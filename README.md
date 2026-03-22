# SIOAR

FastAPI app for food expiry risk prediction, user login, and NGO matching.
The app takes grocery item data, predicts expiry risk, and stores predictions. It also matches items with NGOs that accept the item category.

## Architecture

- `main.py` handles the API routes and login.
- `data_processor.py` cleans the CSV data and prepares training features.
- `ml_engine.py` trains the model and makes predictions.
- `database.py` stores users, predictions, and NGO data in MongoDB.

## Flow

Raw CSV data -> processed training data -> trained model -> API prediction -> MongoDB storage

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Main Routes

- `/auth/signup`
- `/auth/signin`
- `/train`
- `/predict`
- `/ngos`
- `/match`

Run `/train` before `/predict`.
