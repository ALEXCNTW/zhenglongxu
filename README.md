>>>>>>>>>>>Just press start_streamlit.bat!
>>>>>>>>>>>Any problem, call Alex


Our program is designed to predict soccer's goal. 
First, user need to choose a future game.
Second, i have two methods, user need to choose one of them. Automatic is to take real odds from internet automaticlly, using odds api. But this api is not always working, thus, user can also choose Manual. For manual, user can input odds, decided by user himself. 
Third, using these odds, system will calcuate the result. It can predict win/draw/lose, and over/under 2.5 balls. 
In the mean time, it will give user two imgs, one is the reslut between two teams in the last 3 seasons, another is the total goals in the last 3 seasons. 

#pandas:Load .csv data (match results, odds history), clean and preprocess datasets

#numpy:Transform data arrays, save processed datasets to .npz

#joblib: Save and load preprocessing scalers, trained model weights efficiently

#matplotlib:Draw img for history game result

#seaborn:make imgs more beautiful

#streamlit:Our UI system

#PyTorch:Define FNO models, train models, make prediction 

#scikit-learn:StandardScaler for feature scaling, train-test splitting, accuracy calculation

#nest_asyncio:used to solve some problems in streamlib.py

ZhenglongXu(Alex) 
ShipengRen
