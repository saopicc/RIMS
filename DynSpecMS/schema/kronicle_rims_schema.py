import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from kronicle_sdk.conf.read_conf import Settings
from kronicle_sdk.connectors.abc_connector import KroniclePayload
from kronicle_sdk.connectors.channel.channel_setup import KronicleSetup
from kronicle_sdk.models.data.kronicable_sample import KronicableSample
from kronicle_sdk.utils.log import log_d
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


# if __name__ == "__main__":  # pragma: no-cover
#     from datetime import datetime, timezone

#     here = "rims_paylaod"
#     sample_payload = {
#         "name": "Test Pulsar",
#         "tags": ["fast radio burst", "test"],
#         "type": "pulsar",
#         "ra_deg": 123.456,
#         "dec_deg": -22.5,
#         "added_by": IdentifiedPerson(email="omartine@irisa.fr"),
#         "dataset_id": "MS12345",
#         "instrument_name": "MeerKAT",
#         "RIMS client version": "v1.0.0",
#         "data dimensions": {
#             "time_start_utc": datetime.now(timezone.utc).isoformat(),
#             "time_end_utc": (datetime.now(timezone.utc)).isoformat(),
#             "time_resolution_s": 1.0,
#             "frequency_min_mhz": 100.0,
#             "frequency_max_mhz": 200.0,
#             "frequency_resolution_khz": 10.0,
#             "stokes": ["I", "Q", "U", "V"],
#         },
#         "batch access policy": {"visibility": "public", "embargo_months": 0},
#         "products_uri": [
#             "http://example.com/product1.fits",
#             "http://example.com/product2.fits",
#         ],
#         "orcid": "0000-0001-2345-6789",  # optional
#     }

#     try:
#         obs = RimsBatch.model_validate(sample_payload)
#         log_d(here, "ObservationPayload parsed successfully!", obs)
#         # log_d(here, obs.model_dump_json(indent=2, exclude_none=True))  # JSON output using snake_case internally
#         # log_d(here, "obs.channel_schema:", obs.channel_schema)
#     except Exception as e:
#         log_d(here, f"Validation failed: {e}")
#         raise
#     co = Settings().connection
#     kronicle_writer = KronicleWriter(co.url, co.usr, co.pwd)

#     channel_id = "ab0508ea-1312-4b7b-8da1-d6ecd9238284"
#     payload = {
#         "channel_id": channel_id,
#         "channel_name": "RIMS test 4",
#         "channel_schema": obs.channel_schema,
#         "metadata": {"description": RimsBatch.get_field_descriptions()},
#         "tags": {"test": True},
#         "rows": [obs.to_row()],
#     }

# desc_snake = obs.get_field_descriptions()
# log_d(here, "desc_snake", desc_snake)
# log_d(here, "payload", payload)
# result = kronicle_writer.insert_rows_and_upsert_channel(payload)
# log_d(here, "result", result)
# log_d(here, "channels", kronicle_writer.get_all_channels(should_log=True))
# log_d(here, "channels", kronicle_writer.get_channel(id=channel_id))
# log_d(here, "channels", kronicle_writer.get_rows_for_channel(id=channel_id))

if __name__ == "__main__":  # pragma: no-cover
    from datetime import datetime, timezone

    here = "rims_payload_test"

    # --- Build sample subcomponents ---
    source = RimsSource(
        observer=IdentifiedPerson(email="omartine@irisa.fr"),
        dataset_id="MS12345",
        instrument_name="MeerKAT",
    )

    app = AppService(
        hash="v1.0.0",  # type: ignore[arg-type]
        maintainer=IdentifiedPerson(email="maintainer@kronicle.org"),
        computing_infrastructure="SURF",
    )

    batch = RimsBatch(
        tags=[],
        # tags=["fast radio burst", "test"],
        publisher=IdentifiedPerson(email="owner@kronicle.org"),
        data_dimensions=DataDimensions(  # type: ignore[arg-type]
            time_start_utc=datetime.now(timezone.utc),
            time_end_utc=datetime.now(timezone.utc),
            time_resolution_s=1.0,
            frequency_min_mhz=100.0,
            frequency_max_mhz=200.0,
            frequency_resolution_khz=10.0,
            stokes=["I", "Q", "U", "V"],
        ),
        batch_access_policy=AccessPolicy(visibility="public", embargo_months=0),  # type: ignore[arg-type]
    )

    product = RimsProduct(  # type: ignore[arg-type]
        name="Test Pulsar",
        uri="http://example.com/product1.fits",
        ra_deg=123.456,
        dec_deg=-22.5,
        source_type="pulsar",  # type: ignore[arg-type]
        mime="application/fits",
    )

    # --- Instantiate aggregated observation payload ---
    obs_payload = RimsObservationPayload(
        source=source,
        app=app,
        batch=batch,
        product=product,
    )

    # --- Test channel_schema aggregation ---
    aggregated_schema = obs_payload.channel_schema
    log_d(here, "Aggregated channel_schema", aggregated_schema)

    # --- Test field descriptions aggregation ---
    field_descriptions = obs_payload.get_field_descriptions()
    log_d(here, "Aggregated field descriptions", field_descriptions)

    # --- Test row serialization ---
    row = obs_payload.to_row()
    log_d(here, "Serialized row", row)

    # --- Example payload for KronicleWriter ---
    channel_id = "ab0508ea-1312-4b7b-8da1-d6ecd9238284"
    payload = {
        "channel_id": channel_id,
        "channel_name": "RIMS test 4",
        "channel_schema": aggregated_schema,
        "metadata": {"description": field_descriptions},
        "tags": {"test": True},
        "rows": [row],
    }

    log_d(here, "Final payload for KronicleWriter", payload)

    # channel_id = "ab0508ea-1312-4b7b-8da1-d6ecd9238284"

    # co = Settings(ini_file="").connection
    # kronicle_writer = KronicleWriter(co.url, co.usr, co.pwd)
    # payload = KroniclePayload.from_json({
    #     # "channel_id": channel_id,
    #     "channel_name": "RIMS batch info",
    #     "channel_schema": obs_payload.channel_schema,
    #     "metadata": {"description": obs_payload.get_field_descriptions()},
    #     "tags": {"test": True},
    #     "rows": [obs_payload.to_row()],
    # })
    # # result = kronicle_writer.insert_rows_and_upsert_channel(payload)
    # # print(result)
    co = Settings().connection
    kronicle_setup = KronicleSetup(co.url, co.usr, co.pwd)

    channel_id = "ab0508ea-1312-4b7b-8da1-d6ecd9238284"
    # payload = {
    #     "channel_id": channel_id,
    #     "channel_name": "RIMS test 4",
    #     "channel_schema": obs.channel_schema,
    #     "metadata": {"description": RimsBatch.get_field_descriptions()},
    #     "tags": {"test": True},
    #     "rows": [obs.to_row()],
    # }

    # desc_snake = obs.get_field_descriptions()
    # log_d(here, "desc_snake", desc_snake)
    # log_d(here, "payload", payload)
    # result = kronicle_writer.insert_rows_and_upsert_channel(payload)
    # log_d(here, "result", result)
    # log_d(here, "channels", kronicle_writer.get_all_channels(should_log=True))
    # log_d(here, "channels", kronicle_writer.get_channel(id=channel_id))
    # log_d(here, "channels", kronicle_writer.get_rows_for_channel(id=channel_id))

    payload_json = {
        "channel_id": "bf88c5a1-6c6a-4766-b7c6-7c05b44702ec",
        "channel_name": "RIMS network",
        "channel_schema": {
            "instrument_name": "str",
            "dataset_id": "str",
            "observer": "optional[dict]",
            "hash": "str",
            "maintainer": "optional[dict]",
            "computing_infrastructure": "optional[str]",
            "tags": "list",
            "publisher": "dict",
            "catalog_key": "optional[str]",
            "catalog_name": "optional[str]",
            "data_dimensions": "dict",
            "publication_details": "optional[str]",
            "batch_access_policy": "dict",
            "name": "str",
            "uri": "str",
            "source_type": "optional[str]",
            "file_extension": "optional[str]",
            "ra_deg": "float",
            "dec_deg": "float",
        },
        "metadata": {
            "description": {
                "instrument_name": "Name of the instrument used for the observation i.e. MeerKAT, LOFAR, etc.",
                "dataset_id": "Unique identifier for the dataset - may just be the measurement set name if unknown",
                "observer": "Identifier/email/name of the person or system adding the data to Kronicle",
                "hash": "Version of the RIMS client used to generate this payload, ideally a commit hash for reproducibility",
                "computing_infrastructure": "Name of the computing infrastructure used for data processing, e.g., 'SURF', 'AWS', 'Google Cloud', etc.",
                "catalog_key": "Target catalog ID, if target is from a known catalog",
                "catalog_name": "Catalog Name, if target is from a known catalog",
                "data_dimensions": "This will be the time, frequency and polarization coverage",
                "publication_details": "Free-form string for BibTeX entry or ORCID ID. Recommended if data result is used in a publication",
                "source_type": "e.g., star, pulsar, or bright source",
                "file_extension": "MIME type of the file",
                "ra_deg": "Right ascension in degrees [0, 360)",
                "dec_deg": "Declination in degrees [-90, 90]",
            }
        },
        "rows": [
            {
                "instrument_name": "MeerKAT",
                "dataset_id": "1666977074",
                "observer": {"name": "Elizabeth Mahony"},
                "hash": "v1.0-202-gb5b7f75",
                "maintainer": {
                    "email": "myburgh.talon@gmail.com",
                    "orcid": "0000-0000-0000-0000",
                    "name": '"Talon Myburgh"',
                },
                "computing_infrastructure": "Linux x86_64",
                "tags": [],
                "publisher": {
                    "email": "myburgh.talon@gmail.com",
                    "orcid": "0000-0003-0804-9362",
                    "name": "Talon Myburgh",
                },
                "data_dimensions": {
                    "time_start_utc": "2022-10-28 18:00:46.178000+00:00",
                    "time_end_utc": "2022-10-28 21:48:40.596000+00:00",
                    "time_resolution_s": 7.895160606130958,
                    "frequency_min_mhz": 544.25732421875,
                    "frequency_max_mhz": 1087.72607421875,
                    "frequency_resolution_khz": 531.25,
                    "stokes": ["I", "Q", "U", "V"],
                },
                "publication_details": "@article{smirnov2025mining,\n  title={Mining the time axis with TRON--II. MeerKAT detects a stellar radio flare from a distant RS CVn candidate},\n  author={Smirnov, Oleg M and Golden, Aaron and Myburgh, Talon and Ngcebetsha, Buntu and Tasse, Cyril and Heywood, Ian and Ramaila, Athanaseus JT and Thompson, Mark A and Kenyon, Jonathan S and Perkins, Simon J and others},\n  journal={Monthly Notices of the Royal Astronomical Society: Letters},\n  volume={538},\n  number={1},\n  pages={L89--L93},\n  year={2025},\n  publisher={Oxford University Press}\n}",
                "batch_access_policy": {"visibility": "private", "embargo_months": 6},
                "name": "dummy:U4",
                "uri": "https://rims.extragalactic.info/downloads/9ec9a5e7-6df8-4f9e-ba32-e293fa58f1d9",
                "source_type": "I",
                "file_extension": "fits",
                "ra_deg": 339.8136958876477,
                "dec_deg": -23.90536179077666,
            }
        ],
    }

    kronp=KroniclePayload.from_json(payload_json)

    # res=kronicle_writer.insert_rows_and_upsert_channel(payload_json)
    # print(res)
    res = kronicle_setup.get_rows_for_channel("bf88c5a1-6c6a-4766-b7c6-7c05b44702ec")
    # res = kronicle_setup.get_all_channels()
    print()
    print()
    print()
    print(res)
    # print(batch.to_row())

