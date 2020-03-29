# 画像処理プログラム
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input


def examine_cat_breeds(image, model, cat_list):
    # 行列に変換
    img_array = img_to_array(image)
    # 3dim->4dim
    img_dims = np.expand_dims(img_array, axis=0)
    # Predict class（preds：クラスごとの確率が格納された12×1行列）
    preds = model.predict(preprocess_input(img_dims))
    preds_reshape = preds.reshape(-1,preds.shape[0])
    # cat_list(リスト)を12×1行列に変換
    cat_array = np.array(cat_list).reshape(len(cat_list),-1)
    # 確率高い順にソートする
    preds_sort = preds_reshape[np.argsort(preds_reshape[:, 0])[::-1]]
    # 確率の降順に合わせて猫の順番も変える
    cat_sort = cat_array[np.argsort(preds_reshape[:, 0])[::-1]]
    # preds_reshape と cat_arrayを結合
    set_result = np.concatenate([cat_sort, preds_sort], 1)
    return set_result[0:3, :]