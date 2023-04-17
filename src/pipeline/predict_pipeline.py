import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "iplmodel.pkl")
#            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
#            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
#            data_scaled = preprocessor.transform(features)
            self.pred_df = features
            preds = model.predict(self.pred_df)

            teamname = {1: 'Mumbai Indians', 2: 'Chennai Super Kings', 3: 'Kolkata Knight Riders', 4: 'Royal Challengers Bangalore',
                        5: 'Punjab Kings', 6: 'Rajasthan Royals', 7: 'Sunrisers Hyderabad', 8: 'Delhi Capitals', 9: 'Gujarat Titans', 10: 'Lucknow Super Giants'}

            if (preds[0] == int(self.pred_df['team1'][0])) or (preds[0] == int(self.pred_df['team2'][0] )):
                preds = teamname[preds[0]]
                
            else:
                print(
                    f"Prediction is {teamname[preds[0]]}, while Team1: {teamname[self.pred_df['team1'][0]]} and Team2:{teamname[self.pred_df['team2'][0]]}. Prediction defaulted to Team2")
                preds = int(self.pred_df['team2'][0] )
                preds = teamname[preds] + ' '
            
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 year: int,
                 team1: int,
                 team2: int,
                 city: int,
                 toss: int,
                 toss_win: str):

        self.year = year
        self.team1 = team1
        self.team2 = team2
        self.city = city
        self.toss = toss

        self.toss_win = toss_win
        if self.toss_win == 'tm1':
            self.toss_win = team1
        else:
            self.toss_win = team2
        
        # MAP City to Venue internally
        # print(f"city selected is {self.city}")
        venue_dict = {'1':19, '2':16, '6':24, '7':17, '9':10, '10':12, '14':2, '15':25, '17':28, '22':9, '23':3, '24':35, '26':7}
        self.venue = venue_dict[self.city]
        # print(f"venue lookedup is {self.venue}")

# arr = np.array([[season, team1, team2, city, toss, toss_win, venue]]
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "season": [self.year],
                "team1": [self.team1],
                "team2": [self.team2],
                "city": [self.city],
                "toss": [self.toss],
                "toss_win": [self.toss_win],
                "venue": [self.venue]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
