"""The service: health, prediction and the client's two rejection rules."""


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_number(client, valid_request):
    response = client.post("/predict", json=valid_request)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_rejects_impossible_temperature(client, valid_request):
    valid_request["temp_c"] = 75.1
    response = client.post("/predict", json=valid_request)
    assert response.status_code == 422


def test_predict_rejects_impossible_panel_area(client, valid_request):
    valid_request["panel_area_m2"] = 3.9
    response = client.post("/predict", json=valid_request)
    assert response.status_code == 422


def test_predict_rejects_a_missing_field(client, valid_request):
    del valid_request["sunshine_hours"]
    response = client.post("/predict", json=valid_request)
    assert response.status_code == 422
