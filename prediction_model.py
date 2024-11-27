import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the saved model from file
# with open("C:/Users/Kashaf/Desktop/random_forest_model.pkl", 'rb') as file:
#     loaded_model = pickle.load(file)

def loadModel_preprocessAllMaps():
    # Create a gender mapping dictionary
    gender_map = {'F': 0, 'M': 1}

    # Create a Sports mapping dictionary
    sports_list = ['Aeronautics', 'Alpinism', 'Archery', 'Art Competitions', 'Athletics', 'Badminton', 'Baseball', 'Basketball',
                   'Basque Pelota', 'Beach Volleyball', 'Boxing', 'Canoeing', 'Cricket', 'Croquet', 'Cycling', 'Diving',
                   'Equestrianism', 'Fencing', 'Figure Skating', 'Football', 'Golf', 'Gymnastics', 'Handball', 'Hockey',
                   'Ice Hockey', 'Jeu De Paume', 'Judo', 'Lacrosse', 'Modern Pentathlon', 'Motorboating', 'Polo', 'Racquets',
                   'Rhythmic Gymnastics', 'Roque', 'Rowing', 'Rugby', 'Rugby Sevens', 'Sailing', 'Shooting', 'Softball',
                   'Swimming', 'Synchronized Swimming', 'Table Tennis', 'Taekwondo', 'Tennis', 'Trampolining', 'Triathlon',
                   'Tug-Of-War', 'Volleyball', 'Water Polo', 'Weightlifting', 'Wrestling']

    sport_map = {sport: i for i, sport in enumerate(sports_list)}

    # Create a country mapping dictionary
    country_list = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Argentina',
                    'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Barbados',
                    'Belarus', 'Belgium',  'Bermuda', 'Bhutan', 'Boliva', 'Botswana',
                    'Brazil',  'Bulgaria',  'Cambodia', 'Canada', 'Cape Verde',
                    'Cayman Islands', 'Central African Republic', 'Chile', 'China', 'Colombia','Cook Islands',
                    'Costa Rica', 'Croatia', 'Cuba',  'Cyprus', 'Czech Republic',
                    'Denmark', 'Ecuador', 'Egypt','Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gambia', 'Georgia', 'Germany', 'Ghana',
                    'Greece', 'Grenada', 'Guam', 'Guatemala', 'Guinea', 'Guyana',  'Hungary',
                    'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
                    'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kuwait', 'Kyrgyzstan',
                    'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Lithuania', 
                    'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mauritius', 'Mexico',
                    'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
                    'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'Norway', 'Oman', 'Pakistan',
                    'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal',
                    'Qatar', 'Romania', 'Russia', 'Saudi Arabia', 'Senegal', 'Serbia', 
                    'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Korea', 'South Sudan',
                    'Spain', 'Sri Lanka', 'Sudan', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan',
                    'Tanzania', 'Thailand', 'Tunisia', 'Turkey', 'Turkmenistan', 'UK', 'USA',
                    'Uganda', 'Ukraine', 'United Arab Emirates', 'Unknown', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam',
                    'Yemen', 'Zambia', 'Zimbabwe']

    region_map = {region: i for i, region in enumerate(country_list)}

    maps = [gender_map, sport_map, region_map]
    return maps

def predict_athlete_will_win(gender, age, height, weight, sport, country):
    gender_map, sport_map, region_map = loadModel_preprocessAllMaps()

    # Convert string to numerical using map
    gender = gender_map[gender]
    sport = sport_map[sport]
    country = region_map[country]
    input_arr = [gender, age, height, weight, sport, country]

    # Load the model and dataset
    athletepath = 'Encoded_Oversampled_data.csv'
    new_df = pd.read_csv(athletepath)

    # Select features and target
    X = new_df[['Gender', 'Age', 'Height', 'Weight', 'Sport_Encoded', 'Region_Encoded']]
    y = new_df['Medal_Encoded']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the classifier
    # clf = RandomForestClassifier()
    clf = LogisticRegression()

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make prediction for the input data
    input_data = pd.DataFrame([input_arr], columns=['Gender', 'Age', 'Height', 'Weight', 'Sport_Encoded', 'Region_Encoded'])
    prediction = clf.predict(input_data)

    return prediction

def predict_medal_counts(country_name, year):
    # Data Preparation
    country_dataset = pd.read_csv("Country_Medals_Formatted.csv")
    
    # Encode the 'Country_Name' column
    le = LabelEncoder()
    country_dataset['Country_Name'] = le.fit_transform(country_dataset['Country_Name'])

    # Split the data into features and target
    X = country_dataset[['Country_Name', 'Year']]
    y = country_dataset[['Gold', 'Silver', 'Bronze']]

    # Model Selection and Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Regressor
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Prediction
    X_pred = pd.DataFrame([[le.transform([country_name])[0], year]], columns=['Country_Name', 'Year'])
    prediction = model.predict(X_pred)

    # Extract Gold, Silver, Bronze values
    gold = round(prediction[0, 0])
    silver = round(prediction[0, 1])
    bronze = round(prediction[0, 2])

    return gold, silver, bronze