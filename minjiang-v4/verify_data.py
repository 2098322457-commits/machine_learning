"""Verify the local CSV bytes against GitHub's tree at the team's fixed commit."""
import argparse
import hashlib
from pathlib import Path


COMMIT = "498ef586aee5baf18befe75013f7f6ec0815c72f"
GIT_BLOBS = {
    "glofas_daily_discharge.csv": "b4f9851fdf989080629a32920d541d21dafcd7ff",
    "nasa_power_daily.csv": "6bb79353282cde9a8ae74bf938a1a031daeda6a9",
    "openmeteo_hourly_precipitation.csv": "e6ad0138586ae90823a9500ab22dae78bf1706f8",
    "openmeteo_hourly_relative_humidity_2m.csv": "479bd4751ac4823578845db1821cbdb0b62b63e0",
    "openmeteo_hourly_surface_pressure.csv": "9968b04e7ee02cd204777aed46be6bd6e8587647",
    "openmeteo_hourly_temperature_2m.csv": "96bb21098d18b4776afa4479cb035f1f9dc528fb",
    "openmeteo_hourly_wind_speed_10m.csv": "9ebe880d3665ec5f894a382cd7b191f021cd079a",
    "stations.csv": "851076579d6fea3f62f1700f593ff6ea24b97512",
}


def verify(data):
    for name, expected in GIT_BLOBS.items():
        contents = (data / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(contents)}\0".encode("ascii"))
        blob.update(contents)
        if blob.hexdigest() != expected:
            raise ValueError(f"{name} differs from the team's commit {COMMIT}")
    print(f"Verified {len(GIT_BLOBS)} data files against Git commit {COMMIT}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "minjiang-original" / "data")
    verify(parser.parse_args().data)
