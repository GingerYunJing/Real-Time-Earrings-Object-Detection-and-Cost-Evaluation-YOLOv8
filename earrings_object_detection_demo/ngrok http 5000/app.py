# 引入必要的套件
from flask import Flask, render_template, request  # 引入 Flask 相關套件
from werkzeug.utils import secure_filename  # 用於確保上傳的檔案名稱是安全的
from ultralytics import YOLO  # 引入 YOLO 模型
import os
import pyodbc  # 用於與資料庫連接
import shutil  # 用於檔案操作
import uuid  # 用於生成唯一的檔案名稱
import numpy as np
from waitress import serve  # 使用 Waitress 伺服器運行 Flask 應用程式

# 創建 Flask 應用程式
app = Flask(__name__)
index = 0  # 用於追蹤檔案名稱的索引

# 指定 'best.pt' 模型的路徑
model_path = "./best.pt"

# 創建 YOLO 模型
model = YOLO(model_path)

# 定義允許上傳的檔案類型
UPLOAD_FOLDER = './static/'  
# 設定上傳檔案的儲存目錄為相對於應用程式主目錄的 './static/'。
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  
# 將上傳目錄配置設定為 Flask 應用程式的設定中，這樣在其他地方就可以方便地使用 app.config['UPLOAD_FOLDER'] 取得這個目錄。
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}  
# 定義允許上傳的檔案類型，這裡包括 'jpg', 'jpeg', 'png', 和 'gif' 四種擴展名。
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 連接資料庫，使用Windows身份驗證，SERVER=你的電腦名稱
# MSSQL請先建立earring資料庫，並將CSV檔匯入資料庫
connection_database = 'DRIVER={SQL Server};SERVER=HAN;DATABASE=earring;Trusted_Connection=yes;'

# 連接資料庫並獲取資料
def get_data_from_database(predictions):
    try:
        # 連接到資料庫
        connection = pyodbc.connect(connection_database)

        # 創建游標
        cursor = connection.cursor()

        # 執行 SQL 查詢，僅檢索物件偵測結果中的類別名稱在資料庫中存在的相關資料
        query = "SELECT 類別ID, 零件名稱, 庫存數量, 成本價, 廠商ID, 顏色, 材質 FROM parts WHERE 零件名稱 IN (" + ', '.join(f"'{prediction}'" for result in predictions for prediction in result['predictions']) + ")"
        cursor.execute(query)

        # 獲取結果
        rows = cursor.fetchall()

        # 新增查詢 supplier_details 資料表的 SQL 查詢
        supplier_query = "SELECT s.廠商ID, s.廠商名稱, s.廠商電話, s.聯絡人, s.地址, s.email, p.類別ID, p.零件名稱, p.成本價, p.庫存數量 FROM supplier_details s INNER JOIN parts p ON s.廠商ID = p.廠商ID WHERE p.零件名稱 IN (" + ', '.join(f"'{prediction}'" for result in predictions for prediction in result['predictions']) + ") ORDER BY p.類別ID"
        cursor.execute(supplier_query)
        supplier_data = cursor.fetchall()

        return rows, supplier_data

    except pyodbc.Error as ex:
        # 處理連接錯誤
        print(f"連接資料庫時發生錯誤: {ex}")

    finally:
        # 關閉連接
        if connection:
            connection.close()
            print("連接已關閉")

# 計算總數量
def calculate_total_quantity(predictions):
    total_quantity = 0  # 初始化總數量為零

    for result in predictions:  # 遍歷物件偵測結果的預測列表
        total_quantity += len(result['predictions'])  # 將每個預測列表中的物件數量加總

    return total_quantity  # 返回計算得到的總數量


# 計算總成本
def calculate_total_cost(predictions):
    total_cost = 0  # 初始化總成本為零

    for result in predictions:  # 遍歷物件偵測結果的預測列表
        for prediction in result['predictions']:  # 遍歷每個預測結果的預測物件
            cost_query = f"SELECT 成本價 FROM parts WHERE 零件名稱 = '{prediction}'"  # 準備查詢成本價的SQL語句
            try:
                connection = pyodbc.connect(connection_database)  # 連接到資料庫
                cursor = connection.cursor()  # 創建資料庫游標
                cursor.execute(cost_query)  # 執行查詢
                cost = cursor.fetchone()[0]  # 從查詢結果中取得成本價
                total_cost += cost  # 將成本加總到總成本中
            except pyodbc.Error as ex:
                print(f"查詢零件成本時發生錯誤: {ex}")  # 處理資料庫查詢錯誤
            finally:
                if connection:
                    connection.close()  # 確保無論如何都關閉資料庫連接

    return total_cost  # 返回計算得到的總成本


# 處理預測和顯示結果的路由
@app.route('/', methods=['GET', 'POST'])
def predict_and_display():
    # 初始化 class_counts 變數
    class_counts = {}

    if request.method == 'POST':  # 如果是 POST 請求（即使用者上傳檔案）
        if 'image' not in request.files:  # 如果請求中沒有 'image' 檔案
            return render_template('index.html', error='No file part')  # 回傳錯誤訊息到模板

        uploaded_file = request.files['image']  # 從請求中獲取上傳的檔案

        if uploaded_file.filename == '':  # 如果檔案名稱是空的
            return render_template('index.html', error='No selected file')  # 回傳錯誤訊息到模板

        if not allowed_file(uploaded_file.filename):  # 如果檔案擴展名不在允許的擴展名中
            return render_template('index.html', error='Invalid file type')  # 回傳錯誤訊息到模板

        # 儲存上傳的檔案
        new_uuid = uuid.uuid4()
        filename = str(new_uuid) + ".jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 構建儲存檔案的路徑
        uploaded_file.save(file_path)  # 儲存上傳的檔案

        # 進行模型預測
        results = model.predict(file_path, save=True)  # 使用模型進行預測
        names = model.names  # 獲取模型中的類別名稱

        # 辨識結果圖覆蓋原圖(讓前端可以顯示)，並刪除 runs 資料夾(因為第二次使用時路徑會變 predict2...)
        yolodir = ".\\runs\\detect\\predict"
        if index != 0:
            yolodir += str(index)
        yolodir = yolodir + "\\"
        yolodir = yolodir + filename
        shutil.move(yolodir, file_path)  # 移動辨識結果到原始檔案的路徑
        shutil.rmtree("./runs/")  # 刪除運算過程中產生的 runs 資料夾

        # 準備預測結果以便在模板中顯示
        prediction_data = []  # 初始化存儲預測結果的列表

        for result in results:  # 遍歷每一個模型預測的結果
            predictions = []  # 初始化存儲預測物件的列表
            for c in result.boxes.cls:  # 遍歷每個預測框的預測類別
                predictions.append(names[int(c)])  # 將預測類別名稱添加到列表中
                # 更新 class_counts 變數
                class_counts[names[int(c)]] = class_counts.get(names[int(c)], 0) + 1  # 更新 class_counts 變數，統計每個類別出現的次數
            prediction_data.append({'image_path': file_path, 'predictions': predictions})  # 將每個預測結果的路徑和預測物件列表添加到 prediction_data 中

        list_from_tensor = result.boxes.cls.numpy().tolist()  # 將最後一個 result 的預測類別轉換成 Python 列表
        unique_names = np.unique(list_from_tensor)  # 獲取唯一的預測類別
        unique_numbers = len(unique_names)  # 獲取唯一預測類別的數量


        # 從資料庫獲取額外的資料，僅顯示物件偵測結果中存在的相關資料
        database_data, supplier_data = get_data_from_database(prediction_data)

        # 計算總數量和總成本
        total_quantity = calculate_total_quantity(prediction_data)
        total_cost = calculate_total_cost(prediction_data)

        # 將 class_counts、total_quantity、total_cost、database_data 和 supplier_data 傳遞給模板
        return render_template('index.html', predictions=prediction_data, database_data=database_data, supplier_data=supplier_data, class_counts=class_counts, total_quantity=total_quantity, total_cost=total_cost, unique_numbers=unique_numbers)

    # 如果不是 POST 請求，直接返回初始的模板內容
    return render_template('index.html', predictions=None, database_data=None, supplier_data=None, class_counts=class_counts, total_quantity=0, total_cost=0, unique_numbers=0)

# 執行 Flask 應用程式
if __name__ == '__main__':  # 如果這個檔案被直接執行（而不是被引入其他檔案）
    #app.run(debug=True)
    # 使用 Waitress 伺服器運行 Flask 應用程式，用於提供 Flask 應用程式的 HTTP 服務
    serve(app, host="0.0.0.0", port=5000)  # 使用 Waitress 伺服器運行 Flask 應用程式，監聽所有網路介面的 5000 埠口

