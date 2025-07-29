from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow.pyfunc

mlflow.set_tracking_uri("http://13.48.49.105:5000")

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

# Load models at startup
# faq_model = mlflow.pyfunc.load_model("models:/faq-model/Production")
advisor_model = mlflow.pyfunc.load_model("models:/advisor-model/Production")

# @app.post("/faq")
# def ask_faq(payload: Question):
#     answer = faq_model.predict([payload.question])
#     # mlflow pyfunc predict might return list/ndarray; convert to string
#     if isinstance(answer, (list, tuple)):
#         answer = answer[0]
#     return {"response": str(answer)}

# @app.post("/advisor")
# def ask_advisor(payload: Question):
#     answer = advisor_model.predict([payload.question])
#     if isinstance(answer, (list, tuple)):
#         answer = answer[0]
#     return {"response": str(answer)}



@app.post("/advisor")
def ask_advisor(payload: Question):
    answer = advisor_model.predict([payload.question])
    if isinstance(answer, (list, tuple)):
        answer = answer[0]
    
    # If the answer is a dict-like string, parse it
    if isinstance(answer, str) and answer.startswith('{'):
        try:
            import json
            parsed = json.loads(answer.replace("'", '"'))
            return {"response": parsed.get('response', str(answer))}
        except:
            pass
    
    # If it's already a dict
    if isinstance(answer, dict):
        return {"response": answer.get('response', str(answer))}
    
    return {"response": str(answer)}