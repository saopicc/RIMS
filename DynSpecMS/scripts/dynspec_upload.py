import json
import os
import glob
import requests
import numpy as np
from astropy.io import fits
from astropy.time import Time
from datetime import datetime, timezone

from ..schema.kronicle_rims_schema import (
    ObservationPayload, 
    DataDimensions, 
    AccessPolicy,
    RimsProduct,
    RimsSource,
    AppService,
    RimsBatch,
    IdentifiedPerson
)

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

def parse_dynspec_for_metadata(filename: str, run_metadata: dict, visibility: str, embargo_months: int, file_url: str = "") -> ObservationPayload:
    """
    Given a fits filename, parsed run_metadata, and an uploaded file URL, 
    parse the header information and return an ObservationPayload object
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
    
    app_service = AppService(
        **{"RIMS client version": client_version},
        maintainer=IdentifiedPerson(email="maintainer@kronicle.org"),
        computing_infrastructure=sw_meta.get("os_platform", None)
    )
    
    source = RimsSource(
        added_by=IdentifiedPerson(email="community_user@kronicle.org"),
        dataset_id=header.get("OBSID", os.path.basename(filename).replace(".fits", "")).strip(),
        instrument_name=header.get("TEL_NAME", "MeerKAT").strip(),
        data_format="FITS"
    )
    
    batch = RimsBatch(
        name=os.path.basename(run_metadata.get("arguments", {}).get("OutDirName", "unknown_batch")),
        tags=[],
        owner=IdentifiedPerson(email="community_user@kronicle.org"),
        data_dimensions=data_dimensions,
        batch_access_policy=access_policy
    )
    
    product = RimsProduct(
        name=header.get("NAME", "Unknown Target").strip(),
        uri=file_url,
        type=header.get("SRC-TYPE", "Unknown").strip(),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        access_policy=access_policy
    )

    payload = ObservationPayload(
        source=source,
        batch=batch,
        app=app_service,
        product=product
    )
    
    return payload

def publish_to_kronicle(payload: ObservationPayload, kronicle_api_url: str, token: str):
    """
    Template function to publish the parsed ObservationPayload to Kronicle.
    """
    print(f"Publishing {payload} to Kronicle at {kronicle_api_url}...")
    # TODO: Implement actual POST request to Kronicle
    # headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # response = requests.post(kronicle_api_url, headers=headers, data=payload.model_dump_json(by_alias=True))
    # response.raise_for_status()

def process_dynspec_directory(root_dir: str, upload_host: str, upload_token: str, kronicle_api_url: str, kronicle_token: str, visibility: str, embargo_months: int):
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
            try:
                # 1. Upload to host (file server)
                # file_url = file_upload(fits_file, upload_host, upload_token)
                # print(f"Uploaded successfully. URL: {file_url}")
                
                # 2. Parse Metadata generation
                payload = parse_dynspec_for_metadata(
                    fits_file, 
                    run_metadata, 
                    visibility, 
                    embargo_months, 
                    file_url="test.fits"
                )
                
                # 3. Publish to Kronicle
                publish_to_kronicle(payload, kronicle_api_url, kronicle_token)
                print(f"Successfully processed {fits_file}.\n")
                
            except Exception as e:
                print(f"Failed to process {fits_file}. Error: {e}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Upload dynamic spectra FITS files and publish metadata to Kronicle.")
    parser.add_argument("root_dir", help="Root directory containing TARGET, TARGET_W, OFF, OFF_W subdirectories.")
    parser.add_argument("--upload-host", required=True, help="Host URL for file upload.")
    parser.add_argument("--upload-token", required=True, help="Authorization token for file upload.")
    parser.add_argument("--kronicle-api-url", required=True, help="Kronicle API URL for publishing metadata.")
    parser.add_argument("--kronicle-token", required=True, help="Authorization token for Kronicle API.")
    
    parser.add_argument("--visibility", default="public", choices=["public", "private"], help="Visibility of the data (public or private).")
    parser.add_argument("--embargo-months", type=int, default=0, help="Embargo period in months (0-24) after which data becomes public.")
    
    args = parser.parse_args()
    
    if args.embargo_months < 0 or args.embargo_months > 24:
        parser.error("--embargo-months must be between 0 and 24")
    
    process_dynspec_directory(
        root_dir=args.root_dir,
        upload_host=args.upload_host,
        upload_token=args.upload_token,
        kronicle_api_url=args.kronicle_api_url,
        kronicle_token=args.kronicle_token,
        visibility=args.visibility,
        embargo_months=args.embargo_months
    )

if __name__ == "__main__":
    main()