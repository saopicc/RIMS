from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

class DataDimensions(BaseModel):
    time_start_utc: datetime #will automatically convert a unix timestamp or string to a datetime object
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

    @model_validator(mode='after')
    def validate_time_order(self) -> 'DataDimensions':
        if self.time_end_utc <= self.time_start_utc:
            raise ValueError("time_end_utc must be after time_start_utc")
        return self

class BatchAccessPolicy(BaseModel):
    visibility: str = Field(..., description="e.g., 'public' or 'LOFAR KSP'") # can access data whilst still under embargo
    
    embargo_months: int = Field(
        default=0, 
        ge=0, 
        le=24, 
        description="Embargo period in full months (Max 24)"
    )

    @field_validator("embargo_months")
    @classmethod
    def validate_limit(cls, v: int) -> int:
        if v > 24:
            raise ValueError("The maximum allowed embargo is 24 months (2 years).")
        return v
        
class ObservationPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Target Information and origin
    catalog_key: Optional[str] = Field(default=None, description="Target catalog ID, if target is from a known catalog")
    catalog_name: Optional[str] = Field(default=None, description="Catalog Name, if target is from a known catalog")
    name: str
    tags: list[str] = Field(default_factory=list)
    source_type: Optional[str] = Field(None, alias="type", description="e.g., star, pulsar, or bright source")
    
    # Coordinates & Motion
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="Right ascension in degrees [0, 360)")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees [-90, 90]")
    pmra: Optional[float] = Field(
    None, 
    description="Proper motion in RA (mas/yr). Can be positive or negative."
    )
    pmdec: Optional[float] = Field(
        None, 
        description="Proper motion in Dec (mas/yr). Can be positive or negative."
    )
    
    # Data Provenance
    added_by: str = Field(..., default="community", description="Username or identifier of the person or system adding the data to Kronicle") #required
    dataset_id: str = Field(..., description="Unique identifier for the dataset - may just be the measurement set name if unknown") #required
    instrument_name: str = Field(..., description="Name of the instrument used for the observation i.e. MeerKAT, LOFAR, etc.") #required
    computing_infrastructure: Optional[str] = Field(None, description="Name of the computing infrastructure used for data processing, e.g., 'SURF', 'AWS', 'Google Cloud', etc.")
    
    # Publication and Versioning
    publication_details: Optional[str] = Field(
        None, 
        alias="publication details", 
        description="Free-form string for BibTeX entry or ORCID ID. Recommended if data result is used in a publication"
    )
    rims_client_version: str = Field(..., alias="RIMS client version", description="Version of the RIMS client used to generate this payload, ideally a commit hash for reproducibility")

    # Data Characteristics
    data_dimensions: DataDimensions = Field(..., alias="data dimensions", description="This will be the time, frequency and polarization coverage")
    data_format: str = Field(default="FITS")
    
    # Access Policy
    batch_access_policy: BatchAccessPolicy = Field(..., alias="batch access policy")