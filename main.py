from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_DIR = r"C:\Users\KEERTHANA\ml_final_tasks"
models = {
    "linear": pickle.load(open(os.path.join(MODEL_DIR, "Linear_result.pkl"), "rb")),
    "logistic": pickle.load(open(os.path.join(MODEL_DIR, "Logistic_result.pkl"), "rb")),
    "decision": pickle.load(open(os.path.join(MODEL_DIR, "Decisionr_result.pkl"), "rb")),
    "svm": pickle.load(open(os.path.join(MODEL_DIR, "svm_result.pkl"), "rb")),
    "knn": pickle.load(open(os.path.join(MODEL_DIR, "k_nearest.pkl"), "rb")),
    "kmeans": pickle.load(open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "rb")),
    "random": pickle.load(open(os.path.join(MODEL_DIR, "random_forest.pkl"), "rb")),
}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/{model_name}", response_class=HTMLResponse)
def model_page(request: Request, model_name: str):
    return templates.TemplateResponse(f"{model_name}.html", {"request": request})

# --- Prediction routes ---
@app.post("/predict/linear", response_class=HTMLResponse)
def predict_linear(request: Request,
    bedroom: float = Form(...), bathroom: float = Form(...),
    sqft: float = Form(...), sqft_lot: float = Form(...), floors: float = Form(...)
):
    X = [[bedroom, bathroom, sqft, sqft_lot, floors]]
    prediction = models["linear"].predict(X)[0]
    return templates.TemplateResponse("linear.html", {"request": request, "result": f"Predicted House Price: ₹{prediction:,.2f}"})

@app.post("/predict/logistic", response_class=HTMLResponse)
def predict_logistic(request: Request,
    age: int = Form(...), smoker: int = Form(...), ciguret_perday: int = Form(...),
    cholasterol_leval: float = Form(...), bp: float = Form(...), BMI: float = Form(...),
    heart_rate: float = Form(...), glucose: float = Form(...)
):
    X = [[age, smoker, ciguret_perday, cholasterol_leval, bp, BMI, heart_rate, glucose]]
    result = models["logistic"].predict(X)[0]
    msg = "CHD within 10 years" if result == 1 else "No CHD within 10 years"
    return templates.TemplateResponse("logistic.html", {"request": request, "result": msg})

@app.post("/predict/decision", response_class=HTMLResponse)
def predict_decision(request: Request,
    open_price: float = Form(...), high_price: float = Form(...), low_price: float = Form(...)
):
    X = [[open_price, high_price, low_price]]
    prediction = models["decision"].predict(X)[0]
    return templates.TemplateResponse("decision.html", {"request": request, "result": f"Closing Rate: {prediction}"})

@app.post("/predict/svm", response_class=HTMLResponse)
def predict_svm(request: Request,
    radius_mean: float = Form(...), texture_mean: float = Form(...),
    perimeter_mean: float = Form(...), area_mean: float = Form(...), smoothness_mean: float = Form(...)
):
    X = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean]]
    result = models["svm"].predict(X)[0]
    msg = "Malignant tumor" if result == 1 else "Benign tumor"
    return templates.TemplateResponse("svm.html", {"request": request, "result": msg})

@app.post("/predict/knn", response_class=HTMLResponse)
def predict_knn(request: Request,
    volatile_acidity: float = Form(...), citric_acid: float = Form(...),
    density: float = Form(...), pH: float = Form(...),
    sulphates: float = Form(...), alcohol: float = Form(...)
):
    X = [[volatile_acidity, citric_acid, density, pH, sulphates, alcohol]]
    result = models["knn"].predict(X)[0]
    msg = "Good Quality" if result > 6 else "Bad Quality"
    return templates.TemplateResponse("knn.html", {"request": request, "result": msg})

@app.post("/predict/kmeans", response_class=HTMLResponse)
def predict_kmeans(request: Request,
    hair: int = Form(...), feathers: int = Form(...), eggs: int = Form(...), milk: int = Form(...),
    airborne: int = Form(...), aquatic: int = Form(...), predator: int = Form(...), toothed: int = Form(...),
    backbone: int = Form(...), breathes: int = Form(...), venomous: int = Form(...), fins: int = Form(...),
    legs: int = Form(...), tail: int = Form(...), domestic: int = Form(...)
):
    X = [[hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
          backbone, breathes, venomous, fins, legs, tail, domestic]]
    prediction = models["kmeans"].predict(X)[0]
    return templates.TemplateResponse("kmeans.html", {"request": request, "result": f"Animal Class: {prediction}"})

@app.post("/predict/random", response_class=HTMLResponse)
def predict_random(request: Request,
    Gender: int = Form(...), hight: int = Form(...), weight: int = Form(...)
):
    X = [[Gender, hight, weight]]
    prediction = models["random"].predict(X)[0]
    return templates.TemplateResponse("random.html", {"request": request, "result": f"BMI Prediction: {prediction}"})
