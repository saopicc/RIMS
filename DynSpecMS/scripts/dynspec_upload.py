import json
import os
import glob
import requests
import numpy as np
from astropy.io import fits
from astropy.time import Time
from datetime import datetime, timezone

from ..schema.kronicle_rims_schema import (
    RimsObservationPayload, 
    DataDimensions, 
    AccessPolicy,
    RimsProduct,
    RimsSource,
    AppService,
    RimsBatch,
    IdentifiedPerson
)

from kronicle_sdk.models.data.kronicle_payload import KroniclePayload
from kronicle_sdk.connectors.channel.channel_writer import KronicleWriter
from kronicle_sdk.utils.conf_utils import read_ini_conf

def file_upload(filename, host, token):
    """ Upload file filename to host host using authorization token token.
    If successful, returns the URL for the publicly visible uploaded data, else
    raises RuntimeError.
    """
    with open(filename, 'rb') as infile:
        r = requests.post(host, data={'auth':token}, files={'file': infile})
    status = r.json().get('status')
    if status != 'success':
        raise RuntimeError(f'Upload failed with status {status}')
    else:
        return r.json().get('url')

def parse_dynspec_for_metadata(
    filename: str, 
    run_metadata: dict, 
    visibility: str, 
    embargo_months: int, 
    publishing_info: str = None,
    publisher_name: str = None,
    publisher_email: str = None,
    publisher_orcid: str = None,
    maintainer_name: str = None,
    maintainer_email: str = None,
    maintainer_orcid: str = None,
    file_url: str = ""
) -> RimsObservationPayload:
    """
    Given a fits filename, parsed run_metadata, and an uploaded file URL, 
    parse the header information and return an RimsObservationPayload object
    composed of Source, App, Batch, and Product schemas.
    """
    header = fits.getheader(filename)

    # Time parsing
    obs_start = Time(header.get("OBS-STAR"), format="isot", scale="utc").datetime.replace(tzinfo=timezone.utc)
    obs_stop = Time(header.get("OBS-STOP"), format="isot", scale="utc").datetime.replace(tzinfo=timezone.utc)
    
    # Frequency parsing (from FRQ-MIN and FRQ-MAX in Hz)
    freq_min_mhz = header.get("FRQ-MIN", 0.0) / 1e6
    freq_max_mhz = header.get("FRQ-MAX", 0.0) / 1e6
    
    # Frequency resolution (from CHAN-WID in Hz)
    freq_resolution_khz = header.get("CHAN-WID", 0.0) / 1e3
    
    # Time resolution (from CDELT1 in seconds)
    time_resolution_s = abs(header.get("CDELT1", 0.0))

    # Determine stokes parameters present based on NAXIS3
    n_stokes = header.get("NAXIS3", 4)
    stokes_map = ["I", "Q", "U", "V"]
    stokes = stokes_map[:n_stokes]

    data_dimensions = DataDimensions(
        time_start_utc=obs_start,
        time_end_utc=obs_stop,
        time_resolution_s=time_resolution_s,
        frequency_min_mhz=freq_min_mhz,
        frequency_max_mhz=freq_max_mhz,
        frequency_resolution_khz=freq_resolution_khz,
        stokes=stokes
    )

    access_policy = AccessPolicy(
        visibility=visibility,
        embargo_months=embargo_months
    )

    # Mimic inspect_dynspec.py by reading RA_RAD and DEC_RAD and converting to degrees
    ra_rad = header.get("RA_RAD", 0.0)
    dec_rad = header.get("DEC_RAD", 0.0)
    
    ra_deg = float(np.rad2deg(ra_rad)) % 360.0  # modulus ensures [0, 360) range
    dec_deg = float(np.rad2deg(dec_rad))
    dec_deg = max(-90.0, min(90.0, dec_deg))
    
    sw_meta = run_metadata.get("software_metadata", {})
    client_version = sw_meta.get("version", "1.0.0")
    if client_version == "unknown" and sw_meta.get("git_hash") != "Unknown":
        client_version = sw_meta.get("git_hash")

    # Construct the component models
    
    maintainer_person = IdentifiedPerson(
        email=maintainer_email or "maintainer@kronicle.org",
        name=maintainer_name,
        orcid=maintainer_orcid
    )
    
    app_service = AppService(
        **{"RIMS client version": client_version},
        maintainer=maintainer_person,
        computing_infrastructure=sw_meta.get("os_platform", None)
    )
    
    publisher_person = IdentifiedPerson(
        email=publisher_email or "community_user@kronicle.org",
        name=publisher_name,
        orcid=publisher_orcid
    )
    
    source = RimsSource(
        dataset_id=header.get("OBSID", os.path.basename(filename).replace(".fits", "")).strip(),
        instrument_name=header.get("TEL_NAME", "Unknown").strip(),
        observer=IdentifiedPerson(name=header.get("OBSERVER", "Unknown").strip())
    )
    
    batch = RimsBatch(
        name=os.path.basename(run_metadata.get("arguments", {}).get("OutDirName", "unknown_batch")),
        tags=[],
        publisher=publisher_person,
        data_dimensions=data_dimensions,
        batch_access_policy=access_policy,
        **({"publication details": publishing_info} if publishing_info else {})
    )
    
    product = RimsProduct(
        name=header.get("NAME", "Unknown Target").strip(),
        uri=file_url,
        type=header.get("SRC-TYPE", "Unknown").strip(),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        access_policy=access_policy,
        file_extension=filename.split(".")[-1].lower()
    )

    payload = RimsObservationPayload(
        source=source,
        batch=batch,
        app=app_service,
        product=product
    )
    
    return payload

def publish_to_kronicle(payload: RimsObservationPayload, kronicle_user: str, kronicle_pass: str, kronicle_host: str):
    """
    Template function to publish the parsed RimsObservationPayload to Kronicle.
    """
    print(payload.model_dump_json(indent=2))

    kronicle_writer = KronicleWriter(kronicle_host, kronicle_user, kronicle_pass)
    kronicle_payload = {
        "channel_id" : 'bf88c5a1-6c6a-4766-b7c6-7c05b44702ec',
        "channel_name" : "RIMS network",
        "channel_schema" : payload.channel_schema,
        "metadata": {"description": payload.get_field_descriptions()},
        "rows" : [payload.to_row()]
    }
    

    result= kronicle_writer.insert_rows_and_upsert_channel(kronicle_payload)
    print(result)

def process_dynspec_directory(
    root_dir: str, 
    upload_host: str, 
    upload_token: str, 
    kronicle_user: str,
    kronicle_pass: str,
    kronicle_host: str,
    visibility: str, 
    embargo_months: int,
    publishing_info: str = None,
    publisher_name: str = None,
    publisher_email: str = None,
    publisher_orcid: str = None,
    maintainer_name: str = None,
    maintainer_email: str = None,
    maintainer_orcid: str = None
):
    """
    Iterates through TARGET, TARGET_W, OFF, OFF_W directories under root_dir, processing FITS files.
    """
    metadata_file = os.path.join(root_dir, "run_metadata.json")
    run_metadata = {}
    if os.path.isfile(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                run_metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {metadata_file}: {e}")
    else:
        print(f"Warning: No run_metadata.json found in {root_dir}")

    subdirs_to_check = ["TARGET", "TARGET_W", "OFF", "OFF_W"]
    
    for subdir in subdirs_to_check:
        target_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(target_path):
            print(f"Skipping {subdir}: Directory not found.")
            continue
            
        fits_files = glob.glob(os.path.join(target_path, "*.fits"))
        
        for fits_file in fits_files:
            print(f"Processing: {fits_file}")
            # try:
            # 1. Upload to host (file server)
            # file_url = file_upload(fits_file, upload_host, upload_token)
            file_url="https://rims.extragalactic.info/downloads/9ec9a5e7-6df8-4f9e-ba32-e293fa58f1d9"
            print(f"Uploaded successfully. URL: {file_url}")
            
            # 2. Parse Metadata generation
            payload = parse_dynspec_for_metadata(
                fits_file, 
                run_metadata, 
                visibility, 
                embargo_months, 
                publishing_info=publishing_info,
                publisher_name=publisher_name,
                publisher_email=publisher_email,
                publisher_orcid=publisher_orcid,
                maintainer_name=maintainer_name,
                maintainer_email=maintainer_email,
                maintainer_orcid=maintainer_orcid,
                file_url=file_url
            )
            
            # 3. Publish to Kronicle
            publish_to_kronicle(payload, kronicle_user, kronicle_pass, kronicle_host)
            print(f"Successfully processed {fits_file}.\n")
                
            # except Exception as e:
            #     print(f"Failed to process {fits_file}. Error: {e}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Upload dynamic spectra FITS files and publish metadata to Kronicle.")
    parser.add_argument("root_dir", help="Root directory containing TARGET, TARGET_W, OFF, OFF_W subdirectories.")
    parser.add_argument("--server-conf", default="upload_server.ini", help="Path to server configuration INI file.")
    parser.add_argument("--publisher-conf", default="upload_details.ini", help="Path to publisher details INI file.")
    
    args = parser.parse_args()

    server_conf = read_ini_conf(args.server_conf)
    upload_host = server_conf.get("upload", "host")
    upload_token = server_conf.get("upload", "token")
    kronicle_user = server_conf.get("kronicle", "username")
    kronicle_pass = server_conf.get("kronicle", "password")
    kronicle_host = server_conf.get("kronicle", "host")

    print(f"Upload host: {upload_host}"
          f"\nKronicle host: {kronicle_host}"
          f"\nKronicle user: {kronicle_user}"
          f"\nKronicle pass: {kronicle_pass}"
          )

    publisher_conf = read_ini_conf(args.publisher_conf)
    visibility = publisher_conf.get("publishing_details", "visibility", fallback="private")
    embargo_months = publisher_conf.getint("publishing_details", "embargo_months", fallback=0)
    publshing_info = publisher_conf.get("publishing_details", "publishing_info", fallback=None)
    publisher_name = publisher_conf.get("publishing_details", "publisher_name", fallback=None)
    publisher_email = publisher_conf.get("publishing_details", "publisher_email", fallback=None)
    publisher_orcid = publisher_conf.get("publishing_details", "publisher_orcid", fallback=None)
    maintainer_name = publisher_conf.get("publishing_details", "maintainer_name", fallback=None)
    maintainer_email = publisher_conf.get("publishing_details", "maintainer_email", fallback=None)
    maintainer_orcid = publisher_conf.get("publishing_details", "maintainer_orcid", fallback=None)
        
    pub_details = None
    if publshing_info and os.path.isfile(publshing_info):
        with open(publshing_info, 'r') as f:
            pub_details = f.read()
    elif publshing_info:
        print(f"Warning: publishing info file {publshing_info} not found.")
    
    process_dynspec_directory(
        root_dir=args.root_dir,
        upload_host=upload_host,
        upload_token=upload_token,
        kronicle_user=kronicle_user,
        kronicle_pass=kronicle_pass,
        kronicle_host=kronicle_host,
        visibility=visibility,
        embargo_months=embargo_months,
        publishing_info=pub_details,
        publisher_name=publisher_name,
        publisher_email=publisher_email,
        publisher_orcid=publisher_orcid,
        maintainer_name=maintainer_name,
        maintainer_email=maintainer_email,
        maintainer_orcid=maintainer_orcid
    )

if __name__ == "__main__":
    main()