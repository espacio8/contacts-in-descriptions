import logging
import sys
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import RegexpTokenizer
import re
import warnings

warnings.filterwarnings("ignore")


# Определение класса по трансформации данных
class DataPreprocessing:
    def __init__(self):
        # Инициализация логгера для вывода информации
        self.logger = self._get_logger()

        # Компиляция регулярного выражения для поиска телефонных номеров
        self.re_phone = re.compile(
            r"[8-9]\d{7,10}|\+7\d{7,10}|(\d.){8,11}|\+7 \d{3}|8[(-]\d{3}|89 |[8-9] \d|\+7\(|\+7 \(|\d{2}[(-]\d{2}"
        )
        # Компиляция регулярного выражения для поиска мессенджеров
        self.re_messenger = re.compile(
            r"(inst :)|(instagram :)|(vk :)|(discord :)|(телеграм)|(telegram)|(whats app)|(what's app)|(whatsapp)|(вацап)|(вотсап)|(ватсап)|(ват сап)|(viber)|(вайбер)|(тел :)|(телефон :)|(мтс)|(мегафон)"
        )
        # Компиляция регулярного выражения для поиска ссылок (на почту/соц.сети)
        self.re_mail = re.compile(
            r"(http)|(https)|(@mail)|(@gmail)|(@yandex)|(@ya)|(\.com)|(.ru)|(www)"
        )
        # Компиляция регулярного выражения для поиска мобильных номеров
        self.re_mobile = re.compile(
            r"(\+?[7,8].*\d{3}.*\d{3}.*\d{2}.*\d{2})|([9]\d{2}.*\d{3}.*\d{2}.*\d{2})"
        )

        self.pattern = r"""(?x)         # флаг, разрешающий подробные регулярные выражения
        (?:[A-Z]\.)+                    # аббревиатуры, М.Н.К.
      | \w+(?:-\w+)*                    # слова с необязательными внутренними дефисами
      | \$?\d+(?:\.\d+)?%?              # валюта и проценты, $15.50, 100%
      | \.\.\.                          # троеточие
      | [][.,;"'?():_`-]                # разделительные токены, включая ], [
        """
        self.tokenizer = RegexpTokenizer(
            self.pattern
        )  # токенизатор разбивает исходный текст на подстроки, используя регулярное выражение, передаваемое ему в качестве параметра.

        self.stopword = ['и', 'в', 'во', 'что', 'он', 'на', 'я', 'с', 'со',
            'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да',
            'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее',
            'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из',
            'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если',
            'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь',
            'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего',
            'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для',
            'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто',
            'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
            'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
            'ним', 'здесь', 'этом', 'почти', 'мой', 'тем', 'чтобы', 'нее',
            'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
            'наконец', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот',
            'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве',
            'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
            'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда',
            'конечно', 'всю', 'между']

    def remove_stopwords(self, text):
        # Метод для удаления стопслов
        words = [word for word in text if word not in self.stopword]

        return words

    def _get_logger(self):
        # Создание и настройка логгера
        logger = logging.getLogger()
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def remove_punctuation(self, text):
        # Удаление знаков пунктуации из текста
        punctuat = "!\"&'*,;<=>?[\\]^_`{|}~"
        without_punctuat = "".join([c for c in text if c not in punctuat])

        return without_punctuat

    def concat_str(self, x):
        # Соединение слов в строку, разделяя пробелами
        return " ".join([i for i in x])

    def text_clean(self, text):
        # Очистить текст от эмодзи, транспортных знаков, символов карт
        # Компилировать шаблон регулярного выражения
        reg_pattern = re.compile(
            pattern="["
            "\U0001F600-\U0001F64F"  # эмодзи
            "\U0001F680-\U0001F6FF"  # знаки транспорта и карт
            "]+",
            flags=re.UNICODE,
        )
        text = reg_pattern.sub(r"", text)

        return text

    def preprocessing(self, df, name, ordinal_enc=OrdinalEncoder()):
        # Метод для преобразования данных
        self.logger.info(f"Начало предобработки данных в {name}")

        # Преобразование данных datetime
        df.datetime_submitted = pd.to_datetime(df["datetime_submitted"])
        df["hour"] = df["datetime_submitted"].dt.hour

        # Удаление пунктуации, токенизация, удаление стоп-слов, соединение в строки, удаление смайлов, транспортных знаков и обозначений на карте
        self.logger.info(f"Очистка данных в {name}")
        df.description = df.description.apply(lambda x: self.remove_punctuation(x))
        df.description = df.description.apply(
            lambda x: self.tokenizer.tokenize(x.lower())
        )
        df.description = df.description.apply(lambda x: self.remove_stopwords(x))
        df.description = df.description.apply(lambda x: self.concat_str(x))
        df.description = df.description.apply(lambda x: self.text_clean(x))

        # Добавление новых features
        self.logger.info(f"Добавление новых признаков в {name}")
        df["descr_len"] = df.description.apply(lambda x: len(x))
        df["descr_count_7"] = df.description.apply(lambda x: x.count("7"))
        df["descr_count_8"] = df.description.apply(lambda x: x.count("8"))
        df["title_count_7"] = df.title.apply(lambda x: x.count("7"))
        df["title_count_8"] = df.title.apply(lambda x: x.count("8"))
        df["title_count_numb"] = df.title.apply(lambda x: len(re.findall(r"\d", x)))
        df["descr_count_numb"] = df.description.apply(
            lambda x: len(re.findall(r"\d", x))
        )

        df["title_len"] = df.title.apply(
            lambda x: len(x)
        )  # Добавление признака - длины заголовка

        # Бинарные признаки, показывающие наличие в объявлениях и заголовках номеров, мессенджеров, ссылок
        df["re_mobile_descr"] = [
            1 if i == True else 0
            for i in df.description.str.contains(self.re_mobile).fillna(False)
        ]
        df["re_phone_descr"] = [
            1 if i == True else 0
            for i in df.description.str.contains(self.re_phone).fillna(False)
        ]
        df["re_mail_descr"] = [
            1 if i == True else 0
            for i in df.description.str.contains(self.re_mail).fillna(False)
        ]
        df["re_mess_descr"] = [
            1 if i == True else 0
            for i in df.description.str.contains(self.re_messenger).fillna(False)
        ]
        df["re_phone_title"] = [
            1 if i == True else 0
            for i in df.title.str.contains(self.re_phone).fillna(False)
        ]
        df["re_email_title"] = [
            1 if i == True else 0
            for i in df.title.str.contains(self.re_mail).fillna(False)
        ]
        df["messenger"] = df.re_phone_descr & df.re_mess_descr

        df = df.drop(
            ["title", "datetime_submitted"], axis=1
        )  # удаление признаков - заголовка, даты

        # Кодирование категориальных признаков "region", "city"
        self.logger.info(f"Кодирование категориальных признаков в {name}")
        df["region"] = df["region"].apply(lambda x: hash(x) % 10000007)
        df["city"] = df["city"].apply(lambda x: hash(x) % 10000007)

        # Кодирование категориальных признаков "category", "subcategory"
        if name == "train":
            df[["category", "subcategory"]] = ordinal_enc.fit_transform(
                df[["category", "subcategory"]]
            )
        else:
            df[["category", "subcategory"]] = ordinal_enc.transform(
                df[["category", "subcategory"]]
            )

        self.logger.info(f"Конец предобработки данных в {name}")

        return df, ordinal_enc

    def vectorize_svd(
        self,
        df,
        name,
        tf_vectorizer=TfidfVectorizer(max_features=700),
        svd=TruncatedSVD(random_state=42, n_components=70),
    ):
        # Получение векторных представлений для объявлений
        self.logger.info(f"Получение векторных представлений в {name}")
        if name == "train":
            df_descr_tfidf = tf_vectorizer.fit_transform(df.description)
        else:
            df_descr_tfidf = tf_vectorizer.transform(df.description)

        # Уменьшение размерности с помощью усеченного SVD
        self.logger.info(f"Уменьшение размерности разреженной матрицы для {name}")
        if name == "train":
            svd.fit(df_descr_tfidf)
        df_descr_tfidf_svd = pd.DataFrame(svd.transform(df_descr_tfidf), index=df.index)
        X = pd.concat([df, df_descr_tfidf_svd], axis=1)

        return X, tf_vectorizer, svd
