import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
import seaborn as sns 

# Veri setini yükleme
df = pd.read_csv('simge.csv', delimiter=',')


# Eksik değerleri kontrol etme 
print(df.isnull().sum()) 

# Veri setinin başını inceleme 
print(df.head()) 
# Veri setinin genel bilgilerini gözden geçirme 
print(df.info()) 
 
# Eksik değerler doldurma sadece sayısal özellikler için 
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist() 

df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean()) 

# Eksik değerleri tekrar kontrol etme 
print(df.isnull().sum()) 

# Özellikleri ölçeklendirme
scaler = MinMaxScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Bağımsız değişkenler ve hedef değişkeni ayırma
X = df.drop('age_group', axis=1)
y = df['age_group']

# Random Forest modeli oluşturma
rf_model = RandomForestClassifier(random_state=42)

# Eğitim süresini ölçme
start_train_time = time.time()
rf_model.fit(X, y)
end_train_time = time.time()
training_time = end_train_time - start_train_time

# KFold cross-validation için ayarlar
kf = KFold(n_splits=5, shuffle=True, random_state=42)




# F1 score'u metrik olarak kullanarak cross-validation yapma
f1_scorer = make_scorer(f1_score, average='weighted')

# Cross-validation ile model performansını değerlendirme
start_pred_time = time.time()
cv_accuracy = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
cv_f1 = cross_val_score(rf_model, X, y, cv=kf, scoring=f1_scorer)
end_pred_time = time.time()
prediction_time = end_pred_time - start_pred_time

start_cv_time = time.time()
cv_accuracy = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
cv_f1 = cross_val_score(rf_model, X, y, cv=kf, scoring=f1_scorer)
end_cv_time = time.time()
cv_time = end_cv_time - start_cv_time

# Sonuçları gösterme
print("Ortalama Accuracy:", cv_accuracy.mean())
print("Ortalama F-measure:", cv_f1.mean())
print("Eğitim Süresi:", training_time)
print("Tahmin Süresi:", prediction_time)
print("Cross-validation Süresi:", cv_time)

class_counts = df['age_group'].value_counts()

# Pasta grafiği
plt.figure(figsize=(8, 8))
class_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
plt.title('Sınıf Dağılımı - Pasta Grafiği')
plt.ylabel('')
plt.show()


