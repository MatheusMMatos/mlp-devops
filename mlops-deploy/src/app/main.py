# -*- coding: utf-8
import numpy as np
import xgboost as xgb
import pickle
import os

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

# Criação de uma app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

# Habilitando autenticação na app
basic_auth = BasicAuth(app)

# Antes das APIs
colunas = ['score_3',
           'score_4',
           'score_6',
           'risk_rate',
           'credit_limit',
           'income',
           'facebook_profile',
           'n_bankruptcies',
           'n_accounts',
           'n_issues',
           'application_time_in_funnel',
           'external_data_provider_credit_checks_last_month',
           'external_data_provider_email_seen_before',
           'external_data_provider_fraud_score',
           'reported_income',
           'a15',
           'c55',
           'da',
           'dfa',
           'n14',
           'n16',
           'n17',
           'n6',
           'nim',
           'proaty',
           'score_1_0',
           'score_1_1',
           'score_1_2',
           'score_1_3',
           'score_1_4',
           'score_1_5',
           'score_1_6',
           'score_2_1',
           'score_2_7',
           'score_2_8',
           'score_2_9',
           'score_2_11',
           'score_2_12',
           'score_2_14',
           'score_2_15',
           'score_2_16',
           'score_2_17',
           'score_2_19',
           'score_2_21',
           'score_2_23',
           'score_2_24',
           'score_2_26',
           'score_2_27',
           'score_2_30',
           'score_2_34',
           'real_state_1',
           'real_state_3',
           'application_time_applied_1',
           'application_time_applied_2',
           'application_time_applied_3',
           'application_time_applied_5',
           'application_time_applied_9',
           'application_time_applied_22',
           'email_gmail.com',
           'email_outlook.com',
           'marketing_channel_Invite-email',
           'marketing_channel_LinkedIn',
           'marketing_channel_Waiting-list',
           'marketing_channel_Website',
           'regiao_Nordeste',
           'regiao_Sudeste',
           'regiao_Sul']


def load_model(file_name):
    return pickle.load(open(file_name, "rb"))


def load_scaler(file_name):
    return pickle.load(open(file_name, "rb"))


# Carregar modelo treinado
modelo = load_model('../models/logistic_regression_undersampling.pkl')
scaler = load_scaler('../models/scaler_undersampling.sav')


@app.route('/score/', methods=['POST'])
@basic_auth.required
def get_score():
    dados = request.get_json()
    payload = np.array([dados[col] for col in colunas]).reshape(1, -1)
    payload = (scaler.transform(payload))
    score = (modelo.predict_proba(payload))[0]

    status = 'APROVADO' if score[1] <= 0.4 else 'REPROVADO'
    return jsonify(status=status)


@app.route('/score/<cpf>')
@basic_auth.required
def show_cpf(cpf):
    return 'Recebendo dados\nCPF: %s' % cpf


@app.route('/')
def home():
    return 'Funfou'


# Subir a API
app.run(debug=True, host='0.0.0.0')
