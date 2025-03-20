from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/chat/', views.chat_api, name='chat_api'),
    path('patients/', views.patient_list, name='patient_list'),
    path('api/patients/', views.patient_api, name='patient_api'),
    path('api/patients/<int:patient_id>/', views.patient_detail_api, name='patient_detail_api'),
    path('ai-recommendation/', views.ai_recommendation, name='ai_recommendation'),
    path('api/generate-recommendation/', views.generate_recommendation_api, name='generate_recommendation_api'),
    path('api/patients/<int:patient_id>/recommendation/', views.patient_recommendation_api, name='patient_recommendation_api'),
    path('rag-chatbot/', views.rag_chatbot, name='rag_chatbot'),
    path('api/rag-chat/', views.rag_chat_api, name='rag_chat_api'),
    path('api/medical-knowledge-search/', views.medical_knowledge_search_api, name='medical_knowledge_search_api'),
    path('api/risk-assessment/', views.risk_assessment_api, name='risk_assessment_api'),
    path('medical-search/', views.medical_search, name='medical_search'),
] 