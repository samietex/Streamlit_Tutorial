import pickle
from sklearn.preprocessing import LabelEncoder

def load_best_model(filename='best_model.pkl'):
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Best model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"No model file found at {filename}")
        return None
    
def encoder(df, encoders):
    df_encoded = df.copy()
    for col, lb in encoders.items():
        if col in df_encoded.columns:
            df_encoded[col] = lb.transform(df_encoded[col])
    return df_encoded

def load_label_encoders(directory='.'):
    encoders = {}
    for item in os.listdir(directory):
        if item.endswith('_encoder.pkl'):
            col = item.replace('_encoder.pkl', '')
            encoders[col] = joblib.load(os.path.join(directory, item))
    return encoders

def preprocess_input(input_data, encoders):
    for col, encoder in encoders.items():
        if col in input_data:
            # Handling unseen labels: Assign a common category for unseen labels
            known_labels = set(encoder.classes_)
            # Apply lambda function to handle unseen labels
            input_data[col] = input_data[col].apply(lambda x: x if x in known_labels else 'Unseen')
            # Temporary solution: Fit the encoder again on the fly with 'Unseen' label
            # (Note: This is not an ideal solution for a production environment.
            #  A better approach would be adjusting the training phase to anticipate unseen categories.)
            all_labels = np.append(encoder.classes_, 'Unseen')
            encoder.fit(all_labels)
            # Transform the column with the adjusted encoder
            input_data[col] = encoder.transform(input_data[col])
    return input_data


from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

def scaler(df):
    for i in df.columns:
        df[[i]] = mm.fit_transform(df[[i]])
    return df

'''def preprocess_input(input_data, encoders):
    for i in input_data.columns:
        if input_data[i].dtype == 'O':
            # Use the same encoder used during training
            lb = encoders.get(i, LabelEncoder())

            # Fit and transform the label encoder on the current column
            input_data[i] = lb.fit_transform(input_data[i])

            # Save the encoder for future use
            encoders[i] = lb

    # Save updated encoders for future use
    save_label_encoders(encoders)

    return input_data'''

st.title("Profit Prediction")

# Input features based on the dataset
col1, col2 = st.columns((2))

with col1:
    shipmode = st.selectbox('Ship Mode', ['Second Class', 'Standard Class', 'First Class', 'Same Day'])
    segment = st.selectbox('Segment', ['Consumer', 'Corporate', 'Home Office'])
    country = st.selectbox('Country', ['United States'])
    state = st.selectbox('State', ['Kentucky', 'California', 'Florida', 'North Carolina', 'Washington', 'Texas',
                        'Wisconsin', 'Utah', 'Nebraska', 'Pennsylvania', 'Illinois', 'Minnesota',
                        'Michigan', 'Delaware', 'Indiana', 'New York', 'Arizona', 'Virginia',
                        'Tennessee', 'Alabama', 'South Carolina', 'Oregon', 'Colorado', 'Iowa', 'Ohio',
                        'Missouri', 'Oklahoma', 'New Mexico', 'Louisiana', 'Connecticut', 'New Jersey',
                        'Massachusetts', 'Georgia', 'Nevada', 'Rhode Island', 'Mississippi',
                        'Arkansas', 'Montana', 'New Hampshire', 'Maryland', 'District of Columbia',
                        'Kansas', 'Vermont', 'Maine', 'South Dakota', 'Idaho', 'North Dakota',
                        'Wyoming', 'West Virginia'])
    region = st.selectbox('Region', ['South', 'West', 'Central', 'East'])

with col2:
    category = st.selectbox('Category', ['Furniture', 'Office Supplies', 'Technology'])
    sub_category = st.selectbox('Sub-Category', ['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage', 'Furnishings', 'Art',
                                'Phones', 'Binders', 'Appliances', 'Paper', 'Accessories', 'Envelopes',
                                'Fasteners', 'Supplies', 'Machines', 'Copiers'])
    sales = st.slider('Sales', min_value=0, max_value=30000, step=1)
    quantity = st.slider('Quantity', min_value=1, max_value=20, step=1)
    discount = st.slider('Discount', min_value=0, max_value=1)
    month = st.selectbox('Month', ['November', 'June', 'October', 'April', 'December', 'May', 'August', 'July',
                        'September', 'January', 'March', 'February'])

if st.button('Predict Profit'):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame([[shipmode, segment, country, state, region, category, sub_category, sales, quantity, discount, month]],
                              columns=['Ship Mode', 'Segment', 'Country', 'State', 'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Month'])

    st.write('Raw input data:', input_data)

    # Load encoders
    encoders = load_label_encoders()
    

    # Preprocess the input data using loaded encoders
    input_data_preprocessed = encoder(input_data, encoders)

    # Display processed input data
    st.write('Processed input data:', input_data_preprocessed)

    model = load_best_model()

    predictions = model.predict(input_data_preprocessed)[0]

    st.write(f'The predicted profit is {predictions}')