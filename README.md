🥦 Food Expiry Management & Recipe Suggestion 🍽️
This Streamlit-based web app helps you manage food expiry dates and suggests recipes based on items that are expiring soon.

🚀 Features
📂 Upload CSV containing food expiry details.
🗓 Track Expiry Dates and filter out expired items.
📊 Clustering Analysis using K-Means to categorize food items based on expiry dates.
🍽 Recipe Suggestions for items expiring soon.
📑 CSV Format
Ensure your CSV file follows this format:

Food Item	Expiry Date
Milk	2025-03-10
Bread	2025-03-08
Eggs	2025-03-12
🛑 Expiry Date must be in YYYY-MM-DD format.

🏗 Installation & Usage
Clone the Repository
sh
Copy
Edit
git clone https://github.com/your-repo-name.git
cd your-repo-name
Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt
Run the App
sh
Copy
Edit
streamlit run app.py
🛠 Dependencies
Python 3.x
Streamlit
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
📜 License
This project is licensed under the MIT License.
