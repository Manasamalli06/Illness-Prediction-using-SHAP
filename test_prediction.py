import joblib, os, numpy as np, tensorflow as tf
MODEL_PATH = os.path.join('models','illness_risk_model.keras')
scaler = joblib.load(os.path.join('models','scaler.pkl'))
le = joblib.load(os.path.join('models','label_encoder.pkl'))
feature_names = joblib.load(os.path.join('models','feature_names.pkl'))
model = tf.keras.models.load_model(MODEL_PATH)
# Low-risk sample
sample = {'Age':30,'Gender':0,'BMI':22.0,'Systolic_BP':115.0,'Glucose':90.0,'Body_Temp':98.6}
# create array in feature_names order
x = np.array([[sample[name] for name in feature_names]])
x_scaled = scaler.transform(x)
prob = model.predict(x_scaled)[0][0]
pred_class = 1 if prob>0.5 else 0
label = le.inverse_transform([pred_class])[0]
print('SAMPLE:', sample)
print('PROB:', prob)
print('PRED_CLASS:', pred_class, 'LABEL:', label)
