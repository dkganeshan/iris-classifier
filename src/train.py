from sklearn.datasets import load_iris # Load the data
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)

# Split into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
# Train (fit) the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

# Evaluate the model 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)

# Generate the confusion matrix and save it in the output folder
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M") # Create a timestamp for the filename (e.g., 2025-10-20_1530)
output_dir = Path("C:/Users/dkgan/venv/iris-classifier/outputs") # Define the output folder path
output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder if it doesn't exist
save_path = output_dir / f"confusion_matrix_{timestamp}.png"
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(save_path, bbox_inches='tight', dpi=300) # Save the figure
plt.close()

print(f"✅ Confusion matrix saved successfully at: {save_path.resolve()}")