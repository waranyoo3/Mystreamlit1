{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DzZ6jQR8e55t"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "from sklearn import metrics\n",
        "from sklearn.metrics import accuracy_score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wdukv1vRdCEP"
      },
      "outputs": [],
      "source": [
        "df = pd.read_csv('/content/diabetes.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "wLQuox9AdUvg"
      },
      "outputs": [],
      "source": [
        "df.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "uYqEFQ32goS-"
      },
      "outputs": [],
      "source": [
        "df.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "j3wMiJ3x2elr"
      },
      "outputs": [],
      "source": [
        "df.info()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VF_Q5BAz2gem"
      },
      "outputs": [],
      "source": [
        "df.describe()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tH247hUJdW_P"
      },
      "outputs": [],
      "source": [
        "x = df.iloc[:, 0:8]\n",
        "y = df['Outcome']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nsylMK3bd3K1"
      },
      "outputs": [],
      "source": [
        "x.head()\n",
        "y.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Q6WrrmnXd5FB"
      },
      "outputs": [],
      "source": [
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OhbVxt_NmOFF",
        "outputId": "b448186b-25d6-4d0b-8bc2-5c0e74666934"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(614, 8)"
            ]
          },
          "execution_count": 128,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "x_train.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fX-zKCsV6R1Q"
      },
      "outputs": [],
      "source": [
        "sc = StandardScaler()\n",
        "x_train=sc.fit_transform(x_train)\n",
        "x_test=sc.transform(x_test)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NSP2TkY8LAfB"
      },
      "source": [
        "**Decision Tree**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mrdQFqjZg2cZ"
      },
      "outputs": [],
      "source": [
        "DTmodel = DecisionTreeClassifier(criterion='gini')\n",
        "DTmodel.fit(x_train, y_train)\n",
        "\n",
        "y_pred_dt = DTmodel.predict(x_test)\n",
        "accuracy_dt = accuracy_score(y_test, y_pred_dt)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XNIJ3HDnLHlQ"
      },
      "source": [
        "**Naive Bayes**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CkRAUvRsi2z6"
      },
      "outputs": [],
      "source": [
        "NBmodel = GaussianNB()\n",
        "NBmodel.fit(x_train, y_train)\n",
        "\n",
        "y_pred_nb = NBmodel.predict(x_test)\n",
        "accuracy_nb = accuracy_score(y_test, y_pred_nb)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RsQY9qsOLRuC"
      },
      "source": [
        "**KNN**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NBvsRQzlKvNe"
      },
      "outputs": [],
      "source": [
        "KNNmodel = KNeighborsClassifier(n_neighbors=3)\n",
        "KNNmodel.fit(x_train, y_train)\n",
        "\n",
        "y_pred_knn = KNNmodel.predict(x_test)\n",
        "accuracy_knn = accuracy_score(y_test, y_pred_knn)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YT_svgomLbnl"
      },
      "source": [
        "**Accuracy**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MeQhRkXsj17M"
      },
      "outputs": [],
      "source": [
        "print(\"ความแม่นยำของ Decision Tree Classifier:\", '{:.2f}'.format(accuracy_dt))\n",
        "\n",
        "print(\"ความแม่นยำของ Naive Bayes Tree Classifier:\", '{:.2f}'.format(accuracy_nb))\n",
        "\n",
        "print(\"ความแม่นยำของ K-Nearest Neighbors (KNN) Classifier:\", '{:.2f}'.format(accuracy_knn))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "A4FFmstt8zSC"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
