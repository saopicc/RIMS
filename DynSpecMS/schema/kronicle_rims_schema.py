import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from kronicle_sdk.models.data.kronicable_sample import KronicableSample
from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


class DataDimensions(BaseModel):
    time_start_utc: datetime  # will automatically convert a unix timestamp or string to a datetime object
    time_end_utc: datetime
    time_resolution_s: float = Field(
        ..., ge=0, description="Time resolution (t delta) in seconds"
    )
    frequency_min_mhz: float = Field(..., ge=0, description="Minimum frequency in MHz")
    frequency_max_mhz: float = Field(..., ge=0, description="Maximum frequency in MHz")
    frequency_resolution_khz: float = Field(
        ..., ge=0, description="Frequency resolution (channel width) in kHz"
    )

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


class AccessPolicy(BaseModel):
    visibility: str = Field(
        ..., description="e.g., 'public' or 'LOFAR KSP'"
    )  # can access data whilst still under embargo

    embargo_months: int = Field(
        default=0,
        ge=0,
        le=24,
        description="Embargo period in full months (max 24 months)",
    )


class IdentifiedPerson(BaseModel):
    email: Optional[EmailStr] = None
    orcid: Optional[str] = None
    name: Optional[str] = None

    @field_validator("orcid")
    @classmethod
    def validate_orcid(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip(' "\'`') is None:
            return None
        v = v.strip(' "\'`')

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
        v = v.strip(' "\'`').lower()
        return v

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        v = v.strip(' "\'`')
        return v

    def model_dump(self, **params):
        d = super().model_dump(**params)
        return {k: v for k, v in d.items() if v is not None}

class RimsSource(KronicableSample):
    """
    Details about the dataset used as a source of the computation
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=None)

    instrument_name: str = Field(
        ...,
        description="Name of the instrument used for the observation i.e. MeerKAT, LOFAR, etc.",
    )  # required
    dataset_id: str = Field(
        ...,
        description="Unique identifier for the dataset - may just be the measurement set name if unknown",
    )  # required
    observer: Optional[IdentifiedPerson] = Field(
        default=IdentifiedPerson(email="community@kronicle.org"),
        description="Identifier/email/name of the person or system adding the data to Kronicle",
    )
    # data_format: str = Field(default="FITS")


class RimsProduct(KronicableSample):
    """
    One of the files produced by the computation
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=None)

    name: str
    uri: str
    source_type: Optional[str] = Field(
        None, alias="type", description="e.g., star, pulsar, or bright source"
    )
    file_extension: Optional[str] = Field(
        default="application/fits", alias="mime", description="MIME type of the file"
    )
    # Coordinates & Motion
    ra_deg: float = Field(
        ..., ge=0.0, lt=360.0, description="Right ascension in degrees [0, 360)"
    )
    dec_deg: float = Field(
        ..., ge=-90.0, le=90.0, description="Declination in degrees [-90, 90]"
    )
    # pmra: Optional[float] = Field(
    #     None, description="Proper motion in RA (mas/yr). Can be positive or negative."
    # )
    # pmdec: Optional[float] = Field(
    #     None, description="Proper motion in Dec (mas/yr). Can be positive or negative."
    # )


class AppService(KronicableSample):
    """
    Details about the app used for the computation
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=None)

    hash: str = Field(
        ...,
        alias="RIMS client version",
        description="Version of the RIMS client used to generate this payload, ideally a commit hash for reproducibility",
    )
    maintainer: Optional[IdentifiedPerson]

    computing_infrastructure: Optional[str] = Field(
        None,
        description="Name of the computing infrastructure used for data processing, e.g., 'SURF', 'AWS', 'Google Cloud', etc.",
    )


class RimsBatch(KronicableSample):
    """
    Details of the computations
    """

    model_config = ConfigDict(populate_by_name=True, alias_generator=None)

    tags: list[str] = Field(default_factory=list)
    publisher: IdentifiedPerson

    # ----- Data filters
    # Target Information and origin
    catalog_key: Optional[str] = Field(
        default=None, description="Target catalog ID, if target is from a known catalog"
    )
    catalog_name: Optional[str] = Field(
        default=None, description="Catalog Name, if target is from a known catalog"
    )
    # Data Characteristics
    data_dimensions: DataDimensions = Field(
        ...,
        alias="data dimensions",
        description="This will be the time, frequency and polarization coverage",
    )  # and antenna/baseline?

    # ----- Data filters
    # Publication and Versioning
    publication_details: Optional[str] = Field(
        None,
        alias="publication details",
        description="Free-form string for BibTeX entry or ORCID ID. Recommended if data result is used in a publication",
    )

    # Access Policy
    batch_access_policy: AccessPolicy = Field(..., alias="batch access policy")


class RimsObservationPayload(KronicableSample):
    """
    Gathers all the different information about the computation
    """

    source: RimsSource
    app: AppService
    batch: RimsBatch
    product: RimsProduct

    def get_fields(self) -> list[KronicableSample]:
        return [self.source, self.app, self.batch, self.product]

    @classmethod
    def get_field_classes(cls) -> list[type[KronicableSample]]:
        return [RimsSource, AppService, RimsBatch, RimsProduct]

    @classmethod
    def get_all_fields(cls):
        aggregated: dict[str, str] = {}
        for component_cls in cls.get_field_classes():
            if component_cls is not None:
                aggregated.update(component_cls.get_all_fields())
        return aggregated

    @classmethod
    def _get_channel_schema(cls) -> dict[str, str]:
        """
        Aggregate the channel_schema from all subcomponents into a single dict.
        Later keys overwrite earlier keys in case of conflicts.
        """
        aggregated: dict[str, str] = {}
        for component_cls in cls.get_field_classes():
            if component_cls is not None:
                aggregated.update(component_cls._get_channel_schema())
        return aggregated

    @classmethod
    def get_field_descriptions(cls) -> dict[str, str]:
        aggregated: dict[str, str] = {}
        for component in cls.get_field_classes():
            if component is not None:
                aggregated.update(component.get_field_descriptions())
        return aggregated

    def to_row(self) -> dict[str, Any]:
        aggregated: dict[str, Any] = {}
        for component in self.get_fields():
            if component is not None:
                aggregated.update(component.to_row())
        return aggregated


def get_field_descriptions(obj: KronicableSample) -> dict[str, str]:
    """
    Return a dict mapping field names to their description, if a description was provided.
    Works safely for both ModelField and FieldInfo.
    """
    descriptions = {}
    for name, field in obj.__class__.model_fields.items():
        # If it's a ModelField, grab its field_info; else assume it's already FieldInfo
        info = getattr(field, "field_info", field)
        if info.description is not None:
            descriptions[name] = info.description
    return descriptions
