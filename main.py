import pandas as pd
import numpy as np
import joblib
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import datetime
from math import radians, sin, cos, asin, sqrt, atan2, degrees

model = joblib.load("xgboost_nyc_taxi.pkl")
app = FastAPI(title="Taxi Trip Duration Predictor")
geolocator = Nominatim(user_agent="Taxi Trip Predictor")


class TripInput(BaseModel):
    vendor_id: int = Field(..., example=2)
    passenger_count: int = Field(..., example=1, ge=1, le=9)
    store_and_fwd_flag: int = Field(..., example=0, description="0=N, 1=Y")
    pickup_location: str = Field(..., example="Times Square, New York")
    dropoff_location: str = Field(..., example="JFK Airport, New York")
    pickup_datetime: str = Field(..., example="2016-03-14 17:24:55")


def get_coords(location: str):
    try:
        result = geolocator.geocode(location)
        if result:
            return result.latitude, result.longitude
        raise ValueError(f"Location not found: '{location}'")
    except GeocoderTimedOut:
        time.sleep(1)
        return get_coords(location)


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))


def get_bearing(lat1, lon1, lat2, lon2):
    lat1, lat2 = radians(lat1), radians(lat2)
    dlon = radians(lon2 - lon1)
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return degrees(atan2(x, y))


@app.get("/")
def home():
    return {"message": "Taxi Trip Predictor "}


@app.post("/predict")
def predict(trip: TripInput):
    # Geocode
    try:
        plat, plon = get_coords(trip.pickup_location)
        time.sleep(1)
        dlat, dlon = get_coords(trip.dropoff_location)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dt = datetime.strptime(trip.pickup_datetime, "%Y-%m-%d %H:%M:%S")

    features = np.array(
        [
            [
                trip.vendor_id,
                trip.passenger_count,
                plon,
                plat,  # pickup long - lat
                dlon,
                dlat,  # dropoff long lat
                trip.store_and_fwd_flag,
                dt.hour,  # pickup hour
                dt.weekday(),  # pickup dayofweek
                dt.month,  # pickup month
                int(dt.weekday() in [5, 6]),  # pickup is weekend
                haversine(plat, plon, dlat, dlon),  # distance haversine
                abs(plat - dlat) + abs(plon - dlon),  # distance manhattan
                get_bearing(plat, plon, dlat, dlon),  # bearing
            ]
        ]
    )

    # Predict
    duration_sec = float(np.expm1(model.predict(features)[0]))

    return {
        "pickup": {"location": trip.pickup_location, "lat": plat, "lon": plon},
        "dropoff": {"location": trip.dropoff_location, "lat": dlat, "lon": dlon},
        "predicted_duration_seconds": round(duration_sec, 1),
        "predicted_duration_minutes": round(duration_sec / 60, 2),
    }
