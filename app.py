# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from typing import List, Optional
# import uvicorn
# import os
# import shutil
# import tempfile
# import base64

# app = FastAPI()

# # Allow all CORS origins (for testing purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/api/")
# async def analyze(
#     questions: UploadFile = File(...),
#     files: Optional[List[UploadFile]] = File(None)
# ):
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # Save questions.txt
#         questions_path = os.path.join(tmpdir, questions.filename)
#         with open(questions_path, "wb") as f:
#             f.write(await questions.read())

#         # Save other files
#         file_paths = {}
#         if files:
#             for file in files:
#                 file_path = os.path.join(tmpdir, file.filename)
#                 with open(file_path, "wb") as f:
#                     f.write(await file.read())
#                 file_paths[file.filename] = file_path

#         # Process the questions
#         with open(questions_path, 'r') as f:
#             questions_text = f.read()

#         # Placeholder: actual logic to process questions_text and file_paths
#         result = [
#             1,
#             "Titanic",
#             0.485782,
#             "data:image/png;base64," + base64.b64encode(b"fake-image-bytes").decode()
#         ]

#         return JSONResponse(content=result)

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# # To run the application, use the command:
# # uvicorn app:app --host

# 
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
import os
import tempfile
import base64
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from io import BytesIO
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def scrape_wikipedia_table():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="wikitable")
    df = pd.read_html(str(table))[0]
    return df

def make_regression_plot(df):
    df = df.dropna(subset=["Rank", "Peak"])
    df = df[(df["Rank"].apply(lambda x: str(x).isdigit())) & (df["Peak"].apply(lambda x: str(x).isdigit()))]
    df["Rank"] = df["Rank"].astype(int)
    df["Peak"] = df["Peak"].astype(int)

    X = df["Rank"].values.reshape(-1, 1)
    y = df["Peak"].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y)
    plt.plot(X, y_pred, linestyle='dotted', color='red')
    plt.xlabel("Rank")
    plt.ylabel("Peak")
    plt.title("Rank vs Peak with Regression Line")

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    data_uri = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
    return data_uri, float(model.coef_[0][0])

@app.post("/api/")
async def analyze(
    questions: UploadFile = File(...),
    files: Optional[List[UploadFile]] = File(None)
):
    with tempfile.TemporaryDirectory() as tmpdir:
        questions_path = os.path.join(tmpdir, questions.filename)
        with open(questions_path, "wb") as f:
            f.write(await questions.read())

        file_paths = {}
        if files:
            for file in files:
                file_path = os.path.join(tmpdir, file.filename)
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                file_paths[file.filename] = file_path

        with open(questions_path, 'r') as f:
            questions_text = f.read().lower()

        if "highest grossing films" in questions_text:
            df = scrape_wikipedia_table()
            df.columns = [c.replace('\xa0', ' ').strip() for c in df.columns]

            # Q1: $2B before 2000
            count_2b_pre2000 = df[(df["Worldwide gross"].str.contains(r"\$2")) & (df["Year"].astype(str) < '2000')]

            # Q2: earliest > $1.5B
            df["Gross"] = df["Worldwide gross"].str.replace(r"[^\d.]", "", regex=True).astype(float)
            df_over_1_5b = df[df["Gross"] > 1.5e9]
            earliest_title = df_over_1_5b.sort_values("Year").iloc[0]["Title"]

            # Q3 & Q4: Rank vs Peak
            df_clean = df.copy()
            df_clean["Rank"] = df_clean["Rank"].astype(str).str.extract(r"(\d+)")[0]
            df_clean["Peak"] = df_clean["Peak"].astype(str).str.extract(r"(\d+)")[0]
            df_clean = df_clean.dropna(subset=["Rank", "Peak"])
            df_clean["Rank"] = df_clean["Rank"].astype(int)
            df_clean["Peak"] = df_clean["Peak"].astype(int)

            correlation = df_clean["Rank"].corr(df_clean["Peak"])
            plot_uri, slope = make_regression_plot(df_clean)

            result = [
                f"Movies grossing $2B before 2000: {len(count_2b_pre2000)}",
                f"Earliest movie grossing > $1.5B: {earliest_title}",
                f"Correlation between Rank and Peak: {round(correlation, 6)}",
                plot_uri
            ]
        else:
            result = ["Unable to parse the question."]

        return JSONResponse(content=result)

if __name__ == "__main__":
    
    uvicorn.run(app)
