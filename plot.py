import numpy as np
import matplotlib
matplotlib.use(backend="TkAgg")
import matplotlib.pyplot as plt
import process as p
land=p.ndvi>0.0
x_all=p.ndvi[land]
lst=p.celsius.astype('float32')
y_all=lst[land]
sample=5000
pixels=len(x_all)
indices=np.random.choice(pixels,size=sample,replace=False)
plt.figure(figsize=(8,6))
x_sample=x_all[indices]
y_sample=y_all[indices]
plt.scatter(x_sample, y_sample, alpha=0.4, color='crimson', edgecolors='none', s=15)
plt.title("Urban Heat Island Correlation Profile (Ahmedabad)")
plt.xlabel("NDVI (Vegetation Index)")
plt.ylabel("Land Surface Temperature (°C)")
plt.grid(True, linestyle='--', alpha=0.5)
m,c=np.polyfit(x_sample,y_sample,1)
x_line=np.linspace(x_sample.min(),x_sample.max(),100)
y_line=x_line*m+c
plt.plot(x_line, y_line, color='black', linewidth=2, label=f"Trend: {m:.1f}°C per NDVI unit")
plt.legend()
plt.show(block=True)
correlation=np.corrcoef(x_sample,y_sample)
r=correlation[0,1]
r2=r**2
print(f"Slope (m):              {m:.2f}°C per NDVI unit")
print(f"Correlation Coeffecient (r):  {r:.4f}")
print(f"R^2 metric: {r2:.4f}")