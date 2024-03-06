from typing import Tuple, Union
import logging
from lightgbm import LGBMClassifier
import sys
from data_preprocessing import DataPreprocessing
import numpy as np
from sklearn.metrics import roc_auc_score


# Определение класса для обучения модели
class TrainModel:
    def __init__(self) -> None:
        self.logger = self._get_logger()

    def _get_logger(self):
        # Создание и настройка логгера
        logger = logging.getLogger()
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    # Метод класса для обучения модели
    def init_model_train(self, X_train, y_train):
        # Инициализация классификатора
        model = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=700,
            max_depth=12,
            num_leaves=2**12,
            boosting_type="gbdt",
            learning_rate=0.125,
            verbose=-1,
        )
        # Обучение классификатора
        self.logger.info(f"Обучение модели LGBMClassifier")
        model.fit(X_train, y_train)
        self.logger.info(f"Конец обучения модели LGBMClassifier")

        return model

    def pred_val_cat(self, model, X_val):
        # Предсказание значений по категориям объявлений для val
        cats = np.unique(X_val.category.values)
        roc = []
        for cat in cats:
            df = X_val[X_val.category == cat]
            x = df.drop(["description", "is_bad"], axis=1)
            y = df.is_bad
            score = roc_auc_score(y, model.predict_proba(x)[:, 1])
            roc.append(score)

        return sum(roc) / len(roc)


def task1(train_df, val_df, test_df) -> float:
    # Инициализация объекта для предварительной обработки данных
    preprocessor = DataPreprocessing()
    train_df = train_df.dropna()
    val_df = val_df.dropna()
    # Преобразование данных для обучения, валидации и теста
    X_train_pr, train_enc = preprocessor.preprocessing(train_df, name="train")
    X_val_pr, _ = preprocessor.preprocessing(val_df, name="val", ordinal_enc=train_enc)
    X_test_pr, _ = preprocessor.preprocessing(
        test_df, name="test", ordinal_enc=train_enc
    )
    # Векторизация токенов
    X_train_df, train_tf_vectorizer, train_svd = preprocessor.vectorize_svd(
        X_train_pr, name="train"
    )
    X_val_df, _, _ = preprocessor.vectorize_svd(
        X_val_pr, name="val", tf_vectorizer=train_tf_vectorizer, svd=train_svd
    )
    X_test_df, _, _ = preprocessor.vectorize_svd(
        X_test_pr, name="test", tf_vectorizer=train_tf_vectorizer, svd=train_svd
    )

    X_train = X_train_df.drop(["description", "is_bad"], axis=1)
    X_val = X_val_df
    X_test = X_test_df.drop(["description"], axis=1)
    # Извлечение целевых переменных
    y_train = train_df["is_bad"]

    # Инициализация и обучение модели
    trainer = TrainModel()
    model = trainer.init_model_train(X_train, y_train)
    # Вычисление среднего ROC-AUC по категориям объявлений на валидационной выборке
    roc_auc_val = trainer.pred_val_cat(model, X_val)
    print(f"ROC-AUC на валидационной выборке = {roc_auc_val}")
    # Возвращение прогнозов модели для тестовых данных
    trainer.logger.info(f"Предсказание модели LGBMClassifier на тестовой выборке")

    return model.predict_proba(X_test)[:, 1]


def task2(description: str) -> Union[Tuple[int, int], Tuple[None, None]]:
    description_size = len(description)

    if description_size % 2 == 0:
        return None, None
    else:
        return 0, description_size
