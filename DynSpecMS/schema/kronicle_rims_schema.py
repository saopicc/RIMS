import re
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from kronicle_sdk.conf.read_conf import Settings
from kronicle_sdk.connectors.channel.channel_writer import KronicleWriter
from kronicle_sdk.models.data.kronicable_sample import KronicableSample
from kronicle_sdk.utils.log import log_d


class DataDimensions(BaseModel):
    time_start_utc: datetime  # will automatically convert a unix timestamp or string to a datetime object
    time_end_utc: datetime
    time_resolution_s: float = Field(..., ge=0, description="Time resolution (t delta) in seconds")
    frequency_min_mhz: float = Field(..., ge=0, description="Minimum frequency in MHz")
    frequency_max_mhz: float = Field(..., ge=0, description="Maximum frequency in MHz")
    frequency_resolution_khz: float = Field(..., ge=0, description="Frequency resolution (channel width) in kHz")

    # Enforce exact stokes coverage
    stokes: list[Literal["I", "Q", "U", "V"]]

    @field_validator("time_start_utc", "time_end_utc")
    @classmethod
    def check_not_in_future(cls, v: datetime) -> datetime:
        # Check timezone to make sure it does not "look" like it is in the future
        now = datetime.now(timezone.utc)
        if v > now:
            raise ValueError("Observation time cannot be in the future.")
        return v

    @model_validator(mode="after")
    def validate_time_order(self) -> "DataDimensions":
        if self.time_end_utc <= self.time_start_utc:
            raise ValueError("time_end_utc must be after time_start_utc")
        return self


class BatchAccessPolicy(BaseModel):
    visibility: str = Field(
        ..., description="e.g., 'public' or 'LOFAR KSP'"
    )  # can access data whilst still under embargo

    embargo_months: int = Field(default=0, ge=0, le=24, description="Embargo period in full months (max 24 months)")


class RimsUser(BaseModel):
    email: EmailStr
    orcid: Optional[str] = None
    name: Optional[str] = None

    @field_validator("orcid")
    @classmethod
    def validate_orcid(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip() is None:
            return None
        v = v.strip()

        # Match optional https://, optional http://, then orcid.org/,
        # then the 4-digit groups
        match = re.fullmatch(r"(?:https?://)?(?:orcid\.org/)?((\d{4}-){3}\d{4})", v)
        if not match:
            raise ValueError("not a valid ORCID")
        # Return only the 4 quadruplets
        return match[1]

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        v = v.strip().lower()
        return v

    def model_dump(self, **params):
        d = super().model_dump(**params)
        return {k: v for k, v in d.items() if v is not None}


class ObservationPayload(KronicableSample):
    model_config = ConfigDict(populate_by_name=True, alias_generator=None)

    # Target Information and origin
    catalog_key: Optional[str] = Field(default=None, description="Target catalog ID, if target is from a known catalog")
    catalog_name: Optional[str] = Field(default=None, description="Catalog Name, if target is from a known catalog")
    name: str
    tags: list[str] = Field(default_factory=list)
    source_type: Optional[str] = Field(None, alias="type", description="e.g., star, pulsar, or bright source")

    # Coordinates & Motion
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="Right ascension in degrees [0, 360)")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees [-90, 90]")
    pmra: Optional[float] = Field(None, description="Proper motion in RA (mas/yr). Can be positive or negative.")
    pmdec: Optional[float] = Field(None, description="Proper motion in Dec (mas/yr). Can be positive or negative.")

    # Data Provenance
    added_by: RimsUser = Field(
        default=RimsUser(email="community@kronicle.org"),
        description="Identifier/email/name of the person or system adding the data to Kronicle",
    )  # required
    dataset_id: str = Field(
        ..., description="Unique identifier for the dataset - may just be the measurement set name if unknown"
    )  # required
    instrument_name: str = Field(
        ..., description="Name of the instrument used for the observation i.e. MeerKAT, LOFAR, etc."
    )  # required
    computing_infrastructure: Optional[str] = Field(
        None,
        description="Name of the computing infrastructure used for data processing, e.g., 'SURF', 'AWS', 'Google Cloud', etc.",
    )

    # Publication and Versioning
    publication_details: Optional[str] = Field(
        None,
        alias="publication details",
        description="Free-form string for BibTeX entry or ORCID ID. Recommended if data result is used in a publication",
    )
    rims_client_version: str = Field(
        ...,
        alias="RIMS client version",
        description="Version of the RIMS client used to generate this payload, ideally a commit hash for reproducibility",
    )

    # Data Characteristics
    data_dimensions: DataDimensions = Field(
        ..., alias="data dimensions", description="This will be the time, frequency and polarization coverage"
    )  # and antenna/baseline?
    data_format: str = Field(default="FITS")

    # Access Policy
    batch_access_policy: BatchAccessPolicy = Field(..., alias="batch access policy")

    # Products
    products_uri: list[str]

    @classmethod
    def get_field_descriptions(cls) -> dict[str, str]:
        """
        Return a dict mapping field names to their description, if a description was provided.
        Works safely for both ModelField and FieldInfo.
        """
        descriptions = {}
        for name, field in cls.model_fields.items():
            # If it's a ModelField, grab its field_info; else assume it's already FieldInfo
            info = getattr(field, "field_info", field)
            if info.description is not None:
                descriptions[name] = info.description
        return descriptions


if __name__ == "__main__":
    from datetime import datetime, timezone

    here = "rims_paylaod"
    sample_payload = {
        "name": "Test Pulsar",
        "tags": ["fast radio burst", "test"],
        "type": "pulsar",
        "ra_deg": 123.456,
        "dec_deg": -22.5,
        "added_by": RimsUser(email="omartine@irisa.fr"),
        "dataset_id": "MS12345",
        "instrument_name": "MeerKAT",
        "RIMS client version": "v1.0.0",
        "data dimensions": {
            "time_start_utc": datetime.now(timezone.utc).isoformat(),
            "time_end_utc": (datetime.now(timezone.utc)).isoformat(),
            "time_resolution_s": 1.0,
            "frequency_min_mhz": 100.0,
            "frequency_max_mhz": 200.0,
            "frequency_resolution_khz": 10.0,
            "stokes": ["I", "Q", "U", "V"],
        },
        "batch access policy": {"visibility": "public", "embargo_months": 0},
        "products_uri": ["http://example.com/product1.fits", "http://example.com/product2.fits"],
        "orcid": "0000-0001-2345-6789",  # optional
    }

    try:
        obs = ObservationPayload(**sample_payload)
        # log_d(here, "ObservationPayload parsed successfully!")
        # log_d(here, obs.model_dump_json(indent=2, exclude_none=True))  # JSON output using snake_case internally
        # log_d(here, "obs.channel_schema:", obs.channel_schema)
    except Exception as e:
        log_d(here, f"Validation failed: {e}")
        raise
    co = Settings().connection
    kronicle_writer = KronicleWriter(co.url, co.usr, co.pwd)

    channel_id = "ab0508ea-1312-4b7b-8da1-d6ecd9238284"
    payload = {
        "channel_id": channel_id,
        "channel_name": "RIMS test 4",
        "channel_schema": obs.channel_schema,
        "metadata": {"description": ObservationPayload.get_field_descriptions()},
        "tags": {"test": True},
        "rows": [obs.to_row()],
    }

    # desc_snake = obs.get_field_descriptions()
    # log_d(here, "desc_snake", desc_snake)
    # log_d(here, "payload:", payload)
    # result = kronicle_writer.insert_rows_and_upsert_channel(payload)
    # log_d(here, "result", result)
    # log_d(here, "channels", kronicle_writer.get_all_channels(should_log=True))
    log_d(here, "channels", kronicle_writer.get_channel(id=channel_id))
    # log_d(here, "channels", kronicle_writer.get_rows_for_channel(id=channel_id))