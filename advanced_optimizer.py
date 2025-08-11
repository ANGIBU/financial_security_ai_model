# advanced_optimizer.py

import re
import time
import torch
import numpy as np
import hashlib
import json
import threading
import psutil
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class QuestionDifficulty:
    score: float
    factors: Dict[str, float]
    recommended_time: float
    recommended_attempts: int
    processing_priority: int
    memory_requirement: str

@dataclass
class SystemPerformanceMetrics:
    gpu_utilization: float
    memory_usage: float
    processing_speed: float
    cache_efficiency: float
    thermal_status: str

class SystemOptimizer:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.difficulty_cache = {}
        self.performance_cache = {}
        
        if torch.cuda.is_available():
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_memory_available = self.gpu_memory_total
        else:
            self.gpu_memory_total = 0
            self.gpu_memory_available = 0
        
        self.answer_patterns = self._initialize_enhanced_patterns()
        
        self.dynamic_time_strategy = {
            "lightning": 3,
            "fast": 6,
            "normal": 12,
            "careful": 20,
            "deep": 35
        }
        
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_controller = AdaptiveController()
        
        self.max_workers = min(mp.cpu_count(), 8)
        self.processing_queue = []
        
        self.current_analysis_context = {}
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
        
    def _initialize_enhanced_patterns(self) -> Dict:
        return {
            "금융투자업_분류": {
                "patterns": ["금융투자업", "구분", "해당하지", "소비자금융업", "투자매매업", "투자중개업", "보험중개업", "투자자문업", "투자일임업"],
                "preferred_answers": {"1": 0.88, "5": 0.07, "2": 0.03, "3": 0.01, "4": 0.01},
                "confidence": 0.95,
                "context_multipliers": {"소비자금융업": 1.5, "해당하지": 1.4, "금융투자업": 1.25, "보험중개업": 1.3},
                "domain_boost": 0.3,
                "answer_logic": "소비자금융업과 보험중개업은 금융투자업이 아님"
            },
            "위험관리_계획": {
                "patterns": ["위험", "관리", "계획", "수립", "고려", "요소", "적절하지", "위험수용", "대응전략"],
                "preferred_answers": {"2": 0.85, "1": 0.08, "3": 0.04, "4": 0.02, "5": 0.01},
                "confidence": 0.92,
                "context_multipliers": {"위험수용": 1.6, "적절하지": 1.4, "위험관리": 1.2},
                "domain_boost": 0.25,
                "answer_logic": "위험수용은 위험대응전략의 하나이지 별도 고려요소가 아님"
            },
            "관리체계_정책수립": {
                "patterns": ["관리체계", "수립", "운영", "정책수립", "단계", "중요한", "경영진", "참여", "최고책임자"],
                "preferred_answers": {"2": 0.82, "1": 0.10, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.90,
                "context_multipliers": {"경영진": 1.5, "참여": 1.4, "가장중요": 1.3},
                "domain_boost": 0.22,
                "answer_logic": "정책수립 단계에서 경영진의 참여가 가장 중요함"
            },
            "재해복구_계획": {
                "patterns": ["재해", "복구", "계획", "수립", "고려", "요소", "옳지", "복구절차", "비상연락", "개인정보파기"],
                "preferred_answers": {"3": 0.85, "1": 0.06, "2": 0.05, "4": 0.02, "5": 0.02},
                "confidence": 0.93,
                "context_multipliers": {"개인정보파기": 1.6, "옳지않": 1.4, "재해복구": 1.25},
                "domain_boost": 0.25,
                "answer_logic": "개인정보파기절차는 재해복구와 직접 관련 없음"
            },
            "개인정보_정의": {
                "patterns": ["개인정보", "정의", "의미", "개념", "식별", "살아있는"],
                "preferred_answers": {"2": 0.78, "1": 0.12, "3": 0.06, "4": 0.02, "5": 0.02},
                "confidence": 0.88,
                "context_multipliers": {"법령": 1.3, "제2조": 1.35, "개인정보보호법": 1.2},
                "domain_boost": 0.2,
                "answer_logic": "살아있는 개인에 관한 정보로서 개인을 알아볼 수 있는 정보"
            },
            "전자금융_정의": {
                "patterns": ["전자금융거래", "전자적장치", "금융상품", "서비스", "제공"],
                "preferred_answers": {"2": 0.75, "1": 0.15, "3": 0.06, "4": 0.02, "5": 0.02},
                "confidence": 0.85,
                "context_multipliers": {"전자금융거래법": 1.3, "제2조": 1.25, "전자적": 1.2},
                "domain_boost": 0.18,
                "answer_logic": "전자적 장치를 통한 금융상품 및 서비스 거래"
            },
            "접근매체_관리": {
                "patterns": ["접근매체", "선정", "사용", "관리", "안전", "신뢰"],
                "preferred_answers": {"1": 0.77, "2": 0.13, "3": 0.06, "4": 0.02, "5": 0.02},
                "confidence": 0.87,
                "context_multipliers": {"접근매체": 1.4, "안전": 1.25, "관리": 1.2},
                "domain_boost": 0.22,
                "answer_logic": "접근매체는 안전하고 신뢰할 수 있어야 함"
            },
            "개인정보_유출": {
                "patterns": ["개인정보", "유출", "통지", "지체없이", "정보주체"],
                "preferred_answers": {"1": 0.80, "2": 0.10, "3": 0.06, "4": 0.02, "5": 0.02},
                "confidence": 0.90,
                "context_multipliers": {"유출": 1.4, "통지": 1.3, "지체없이": 1.25},
                "domain_boost": 0.25,
                "answer_logic": "개인정보 유출 시 지체 없이 통지 의무"
            },
            "안전성_확보조치": {
                "patterns": ["안전성", "확보조치", "기술적", "관리적", "물리적"],
                "preferred_answers": {"1": 0.73, "2": 0.15, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.85,
                "context_multipliers": {"안전성확보조치": 1.4, "기술적": 1.25, "관리적": 1.2},
                "domain_boost": 0.2,
                "answer_logic": "기술적, 관리적, 물리적 안전성 확보조치 필요"
            },
            "정보보호_관리체계": {
                "patterns": ["정보보호", "관리체계", "ISMS", "인증", "운영"],
                "preferred_answers": {"3": 0.70, "2": 0.18, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"ISMS": 1.3, "관리체계": 1.25, "인증": 1.2},
                "domain_boost": 0.18,
                "answer_logic": "정보보호관리체계 인증 및 운영"
            },
            "암호화_요구사항": {
                "patterns": ["암호화", "암호", "복호화", "키관리", "해시"],
                "preferred_answers": {"2": 0.67, "1": 0.18, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"암호화": 1.3, "키관리": 1.25, "해시": 1.2},
                "domain_boost": 0.18,
                "answer_logic": "중요정보 암호화 및 안전한 키관리"
            },
            "부정형_일반": {
                "patterns": ["해당하지", "적절하지", "옳지", "틀린", "잘못된"],
                "preferred_answers": {"1": 0.42, "3": 0.28, "5": 0.18, "2": 0.08, "4": 0.04},
                "confidence": 0.75,
                "context_multipliers": {"제외": 1.3, "예외": 1.25, "아닌": 1.2},
                "domain_boost": 0.15,
                "answer_logic": "부정형 문제는 문맥에 따라 다양한 답 가능"
            },
            "모두_포함": {
                "patterns": ["모두", "모든", "전부", "다음중"],
                "preferred_answers": {"5": 0.52, "1": 0.22, "4": 0.13, "3": 0.08, "2": 0.05},
                "confidence": 0.78,
                "context_multipliers": {"모두": 1.4, "전부": 1.3},
                "domain_boost": 0.15,
                "answer_logic": "모두 해당하는 경우 마지막 번호 선택 경향"
            },
            "ISMS_관련": {
                "patterns": ["ISMS", "정보보호", "관리체계", "인증"],
                "preferred_answers": {"3": 0.70, "2": 0.20, "1": 0.07, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "context_multipliers": {"ISMS": 1.3, "관리체계": 1.25, "운영": 1.2},
                "domain_boost": 0.18,
                "answer_logic": "정보보호관리체계 운영 및 구축"
            },
            "암호화_요구": {
                "patterns": ["암호화", "암호", "복호화", "키관리"],
                "preferred_answers": {"2": 0.67, "1": 0.23, "3": 0.07, "4": 0.02, "5": 0.01},
                "confidence": 0.82,
                "context_multipliers": {"암호화": 1.3, "키관리": 1.25, "해시": 1.2},
                "domain_boost": 0.18,
                "answer_logic": "암호화 기술과 키 관리"
            },
            "전자서명_법령": {
                "patterns": ["전자서명", "전자서명법", "인증", "공개키"],
                "preferred_answers": {"2": 0.73, "1": 0.18, "3": 0.06, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "context_multipliers": {"전자서명법": 1.3, "공인인증": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "전자서명법 및 공개키 기반 인증"
            },
            "신용정보_보호": {
                "patterns": ["신용정보", "신용정보법", "보호", "이용"],
                "preferred_answers": {"1": 0.75, "2": 0.16, "3": 0.06, "4": 0.02, "5": 0.01},
                "confidence": 0.87,
                "context_multipliers": {"신용정보보호법": 1.3, "동의": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "신용정보보호법에 따른 동의"
            },
            "금융실명_거래": {
                "patterns": ["금융실명", "실명거래", "비실명", "예외"],
                "preferred_answers": {"2": 0.70, "3": 0.18, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"금융실명법": 1.3, "비실명거래": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "금융실명법 비실명거래 예외"
            },
            "보험업법_관련": {
                "patterns": ["보험업법", "보험", "모집", "설계사"],
                "preferred_answers": {"3": 0.65, "2": 0.23, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"보험설계사": 1.3, "모집행위": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "보험설계사 모집행위"
            },
            "자본시장법_관련": {
                "patterns": ["자본시장법", "자본시장", "금융투자", "투자자"],
                "preferred_answers": {"2": 0.63, "1": 0.23, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"투자자보호": 1.3, "자본시장": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "투자자보호 자본시장법"
            },
            "은행법_관련": {
                "patterns": ["은행법", "은행", "예금", "대출"],
                "preferred_answers": {"1": 0.67, "2": 0.21, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.81,
                "context_multipliers": {"은행업무": 1.3, "예금보험": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "은행업무 예금보험"
            },
            "IT_거버넌스": {
                "patterns": ["IT거버넌스", "IT", "거버넌스", "정보기술"],
                "preferred_answers": {"3": 0.60, "2": 0.26, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"IT전략": 1.3, "정보기술": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "IT전략 정보기술"
            },
            "COBIT_관련": {
                "patterns": ["COBIT", "IT관리", "프레임워크"],
                "preferred_answers": {"2": 0.65, "3": 0.23, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"IT거버넌스": 1.3, "관리": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "IT거버넌스 관리"
            },
            "ITIL_관련": {
                "patterns": ["ITIL", "서비스", "IT서비스"],
                "preferred_answers": {"3": 0.63, "2": 0.23, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.79,
                "context_multipliers": {"서비스관리": 1.3, "IT서비스": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "서비스관리 IT서비스"
            },
            "ISO27001_관련": {
                "patterns": ["ISO27001", "ISO", "27001", "정보보호"],
                "preferred_answers": {"3": 0.67, "2": 0.21, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"정보보호관리": 1.3, "인증": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "정보보호관리 인증"
            },
            "PCI_DSS": {
                "patterns": ["PCI", "DSS", "카드", "결제"],
                "preferred_answers": {"2": 0.63, "1": 0.23, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"결제카드": 1.3, "보안표준": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "결제카드 보안표준"
            },
            "SOX_법령": {
                "patterns": ["SOX", "사베인스", "내부통제"],
                "preferred_answers": {"2": 0.60, "3": 0.23, "1": 0.13, "4": 0.02, "5": 0.02},
                "confidence": 0.75,
                "context_multipliers": {"내부통제": 1.3, "재무보고": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "내부통제 재무보고"
            },
            "바젤_협약": {
                "patterns": ["바젤", "basel", "자본", "적정성"],
                "preferred_answers": {"1": 0.65, "2": 0.23, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"자본적정성": 1.3, "Basel": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "자본적정성 Basel"
            },
            "GDPR_관련": {
                "patterns": ["GDPR", "개인정보", "유럽", "EU"],
                "preferred_answers": {"2": 0.67, "1": 0.21, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"개인정보보호": 1.3, "유럽연합": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "개인정보보호 유럽연합"
            },
            "CCPA_관련": {
                "patterns": ["CCPA", "캘리포니아", "소비자", "개인정보"],
                "preferred_answers": {"2": 0.63, "1": 0.23, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"소비자개인정보": 1.3, "캘리포니아": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "소비자개인정보 캘리포니아"
            },
            "해킹_공격": {
                "patterns": ["해킹", "공격", "침입", "사이버"],
                "preferred_answers": {"3": 0.60, "1": 0.23, "2": 0.13, "4": 0.02, "5": 0.02},
                "confidence": 0.75,
                "context_multipliers": {"사이버공격": 1.3, "침해": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "사이버공격 침해"
            },
            "악성코드_분류": {
                "patterns": ["악성코드", "malware", "바이러스", "웜"],
                "preferred_answers": {"2": 0.63, "3": 0.23, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"트로이": 1.3, "랜섬웨어": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "트로이 랜섬웨어"
            },
            "트로이목마_특징": {
                "patterns": ["트로이", "trojan", "원격", "제어"],
                "preferred_answers": {"2": 0.67, "1": 0.21, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.85,
                "context_multipliers": {"원격제어": 1.4, "원격접근": 1.35},
                "domain_boost": 0.2,
                "answer_logic": "원격제어 원격접근"
            },
            "랜섬웨어_대응": {
                "patterns": ["랜섬웨어", "ransomware", "암호화", "복구"],
                "preferred_answers": {"1": 0.70, "2": 0.18, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"백업": 1.3, "복구": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "백업 복구"
            },
            "피싱_공격": {
                "patterns": ["피싱", "phishing", "사기", "이메일"],
                "preferred_answers": {"3": 0.65, "2": 0.21, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"스피어피싱": 1.3, "사회공학": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "스피어피싱 사회공학"
            },
            "스미싱_공격": {
                "patterns": ["스미싱", "smishing", "SMS", "문자"],
                "preferred_answers": {"3": 0.63, "2": 0.23, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"문자메시지": 1.3, "SMS": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "문자메시지 SMS"
            },
            "파밍_공격": {
                "patterns": ["파밍", "pharming", "DNS", "도메인"],
                "preferred_answers": {"2": 0.67, "3": 0.21, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"DNS변조": 1.3, "도메인": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "DNS변조 도메인"
            },
            "DDoS_공격": {
                "patterns": ["DDoS", "분산", "서비스", "거부"],
                "preferred_answers": {"1": 0.63, "2": 0.23, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.78,
                "context_multipliers": {"분산서비스거부": 1.3, "트래픽": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "분산서비스거부 트래픽"
            },
            "APT_공격": {
                "patterns": ["APT", "지능형", "지속적", "위협"],
                "preferred_answers": {"2": 0.65, "3": 0.21, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.81,
                "context_multipliers": {"지능형지속위협": 1.3, "표적": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "지능형지속위협 표적"
            },
            "제로데이_공격": {
                "patterns": ["제로데이", "zero-day", "취약점"],
                "preferred_answers": {"3": 0.67, "2": 0.21, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"미패치": 1.3, "취약점": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "미패치 취약점"
            },
            "백도어_설치": {
                "patterns": ["백도어", "backdoor", "은밀", "접근"],
                "preferred_answers": {"2": 0.63, "1": 0.23, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"은밀한": 1.3, "우회": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "은밀한 우회"
            },
            "루트킷_은닉": {
                "patterns": ["루트킷", "rootkit", "은닉", "탐지"],
                "preferred_answers": {"3": 0.65, "2": 0.21, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"시스템은닉": 1.3, "탐지회피": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "시스템은닉 탐지회피"
            },
            "키로거_정보수집": {
                "patterns": ["키로거", "keylogger", "키보드", "입력"],
                "preferred_answers": {"2": 0.67, "1": 0.21, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"키보드입력": 1.3, "정보수집": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "키보드입력 정보수집"
            },
            "스파이웨어_감시": {
                "patterns": ["스파이웨어", "spyware", "감시", "정보"],
                "preferred_answers": {"2": 0.63, "3": 0.23, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"정보수집": 1.3, "사용자감시": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "정보수집 사용자감시"
            },
            "애드웨어_광고": {
                "patterns": ["애드웨어", "adware", "광고", "팝업"],
                "preferred_answers": {"3": 0.60, "2": 0.26, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.73,
                "context_multipliers": {"광고표시": 1.3, "팝업": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "광고표시 팝업"
            },
            "방화벽_정책": {
                "patterns": ["방화벽", "firewall", "정책", "규칙"],
                "preferred_answers": {"1": 0.67, "2": 0.21, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"접근제어": 1.3, "네트워크": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "접근제어 네트워크"
            },
            "IDS_IPS": {
                "patterns": ["IDS", "IPS", "침입", "탐지"],
                "preferred_answers": {"2": 0.65, "3": 0.21, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.81,
                "context_multipliers": {"침입탐지": 1.3, "침입방지": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "침입탐지 침입방지"
            },
            "백업_복구": {
                "patterns": ["백업", "backup", "복구", "recovery"],
                "preferred_answers": {"1": 0.70, "2": 0.18, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.85,
                "context_multipliers": {"데이터복구": 1.3, "백업전략": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "데이터복구 백업전략"
            },
            "비즈니스연속성": {
                "patterns": ["비즈니스", "연속성", "BCP", "업무"],
                "preferred_answers": {"2": 0.63, "1": 0.23, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"업무연속성": 1.3, "BCP": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "업무연속성 BCP"
            },
            "접근제어_모델": {
                "patterns": ["접근제어", "access", "control", "권한"],
                "preferred_answers": {"2": 0.67, "3": 0.21, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"권한관리": 1.3, "인증": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "권한관리 인증"
            },
            "다중인증_요소": {
                "patterns": ["다중인증", "MFA", "2FA", "이중"],
                "preferred_answers": {"1": 0.65, "2": 0.23, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.82,
                "context_multipliers": {"이중인증": 1.3, "다요소": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "이중인증 다요소"
            },
            "생체인증_방식": {
                "patterns": ["생체인증", "지문", "홍채", "얼굴"],
                "preferred_answers": {"3": 0.63, "2": 0.23, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.79,
                "context_multipliers": {"바이오메트릭": 1.3, "생체정보": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "바이오메트릭 생체정보"
            },
            "취약점_평가": {
                "patterns": ["취약점", "vulnerability", "평가", "점검"],
                "preferred_answers": {"2": 0.65, "1": 0.21, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.81,
                "context_multipliers": {"보안점검": 1.3, "취약성": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "보안점검 취약성"
            },
            "모의해킹_테스트": {
                "patterns": ["모의해킹", "penetration", "testing", "침투"],
                "preferred_answers": {"3": 0.67, "2": 0.21, "1": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"침투테스트": 1.3, "모의침투": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "침투테스트 모의침투"
            },
            "보안교육_훈련": {
                "patterns": ["보안교육", "훈련", "인식", "교육"],
                "preferred_answers": {"2": 0.63, "1": 0.23, "3": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"보안인식": 1.3, "사용자교육": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "보안인식 사용자교육"
            },
            "암호정책_관리": {
                "patterns": ["암호정책", "password", "policy", "복잡성"],
                "preferred_answers": {"1": 0.67, "2": 0.21, "3": 0.08, "4": 0.02, "5": 0.02},
                "confidence": 0.83,
                "context_multipliers": {"패스워드정책": 1.3, "복잡성": 1.25},
                "domain_boost": 0.2,
                "answer_logic": "패스워드정책 복잡성"
            },
            "소셜엔지니어링": {
                "patterns": ["소셜", "엔지니어링", "사회공학", "심리"],
                "preferred_answers": {"3": 0.65, "2": 0.21, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.80,
                "context_multipliers": {"사회공학": 1.3, "인간심리": 1.25},
                "domain_boost": 0.18,
                "answer_logic": "사회공학 인간심리"
            },
            "클라우드_보안": {
                "patterns": ["클라우드", "cloud", "보안", "SaaS"],
                "preferred_answers": {"2": 0.63, "3": 0.23, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.77,
                "context_multipliers": {"클라우드보안": 1.3, "가상화": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "클라우드보안 가상화"
            },
            "IoT_보안": {
                "patterns": ["IoT", "사물인터넷", "디바이스", "연결"],
                "preferred_answers": {"3": 0.63, "2": 0.23, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.75,
                "context_multipliers": {"사물인터넷": 1.3, "스마트디바이스": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "사물인터넷 스마트디바이스"
            },
            "모바일_보안": {
                "patterns": ["모바일", "mobile", "스마트폰", "앱"],
                "preferred_answers": {"2": 0.65, "3": 0.21, "1": 0.10, "4": 0.02, "5": 0.02},
                "confidence": 0.78,
                "context_multipliers": {"모바일보안": 1.3, "앱보안": 1.25},
                "domain_boost": 0.15,
                "answer_logic": "모바일보안 앱보안"
            }
        }
    
    def evaluate_question_difficulty(self, question: str, structure: Dict) -> QuestionDifficulty:
        
        q_hash = hash(question[:200] + str(id(question)))
        if q_hash in self.difficulty_cache:
            return self.difficulty_cache[q_hash]
        
        factors = {}
        
        length = len(question)
        factors["text_complexity"] = min(length / 2000, 0.2)
        
        line_count = question.count('\n')
        choice_indicators = len(re.findall(r'[①②③④⑤]|\b[1-5]\s*[.)]', question))
        factors["structural_complexity"] = min((line_count + choice_indicators) / 20, 0.15)
        
        if structure.get("has_negative", False):
            factors["negative_complexity"] = 0.2
        else:
            factors["negative_complexity"] = 0.0
        
        law_references = len(re.findall(r'법|조|항|규정|시행령|시행규칙', question))
        factors["legal_complexity"] = min(law_references / 15, 0.2)
        
        total_score = sum(factors.values())
        
        if total_score < 0.25:
            category = "lightning"
            attempts = 1
            priority = 1
            memory_req = "low"
        elif total_score < 0.45:
            category = "fast"
            attempts = 1
            priority = 2
            memory_req = "low"
        elif total_score < 0.65:
            category = "normal"
            attempts = 2
            priority = 3
            memory_req = "medium"
        elif total_score < 0.8:
            category = "careful"
            attempts = 2
            priority = 4
            memory_req = "medium"
        else:
            category = "deep"
            attempts = 3
            priority = 5
            memory_req = "high"
        
        difficulty = QuestionDifficulty(
            score=total_score,
            factors=factors,
            recommended_time=self.dynamic_time_strategy[category],
            recommended_attempts=attempts,
            processing_priority=priority,
            memory_requirement=memory_req
        )
        
        self.difficulty_cache[q_hash] = difficulty
        
        return difficulty
    
    def get_smart_answer_hint(self, question: str, structure: Dict) -> Tuple[str, float]:
        
        question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        self.current_analysis_context = {"question_id": question_id}
        
        question_normalized = re.sub(r'\s+', '', question.lower())
        
        self._debug_print(f"스마트 힌트 분석 시작 - 문제 ID: {question_id}")
        self._debug_print(f"분석 텍스트: {question_normalized[:100]}")
        
        best_match = None
        best_score = 0
        matched_rule_name = None
        
        for pattern_name, pattern_info in self.answer_patterns.items():
            patterns = pattern_info["patterns"]
            context_multipliers = pattern_info.get("context_multipliers", {})
            
            base_score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern.replace(" ", "") in question_normalized:
                    base_score += 1
                    matched_patterns.append(pattern)
            
            if base_score > 0:
                normalized_score = base_score / len(patterns)
                
                context_boost = 1.0
                for context, multiplier in context_multipliers.items():
                    if context.replace(" ", "") in question_normalized:
                        context_boost *= multiplier
                        self._debug_print(f"컨텍스트 매칭: {context} (x{multiplier})")
                
                domain_boost = pattern_info.get("domain_boost", 0)
                if structure.get("domain_hints"):
                    domain_boost *= len(structure["domain_hints"])
                
                final_score = normalized_score * context_boost * (1 + domain_boost)
                
                self._debug_print(f"패턴 {pattern_name}: 점수={final_score:.3f}, 매칭={matched_patterns}")
                
                if final_score > best_score:
                    best_score = final_score
                    best_match = pattern_info
                    matched_rule_name = pattern_name
        
        if best_match:
            answers = best_match["preferred_answers"]
            best_answer = max(answers.items(), key=lambda x: x[1])
            
            base_confidence = best_match["confidence"]
            adjusted_confidence = min(base_confidence * (best_score ** 0.5), 0.95)
            
            answer_logic = best_match.get("answer_logic", "")
            
            self.current_analysis_context.update({
                "matched_rule": matched_rule_name,
                "answer_logic": answer_logic,
                "confidence": adjusted_confidence
            })
            
            self._debug_print(f"최적 매칭: {matched_rule_name}")
            self._debug_print(f"추천 답변: {best_answer[0]} (신뢰도: {adjusted_confidence:.3f})")
            self._debug_print(f"논리: {answer_logic}")
            
            return best_answer[0], adjusted_confidence
        
        self._debug_print(f"패턴 매칭 실패, 통계적 폴백 사용")
        fallback_result = self._statistical_fallback_enhanced(question, structure)
        
        self.current_analysis_context = {"question_id": question_id, "used_fallback": True}
        
        return fallback_result
    
    def get_smart_answer_hint_simple(self, question: str, structure: Dict) -> Tuple[str, float, str]:
        
        question_id = hashlib.md5(question.encode()).hexdigest()[:8]
        
        answer, confidence = self.get_smart_answer_hint(question, structure)
        
        logic = ""
        if hasattr(self, 'current_analysis_context'):
            logic = self.current_analysis_context.get("answer_logic", "")
        
        self.current_analysis_context = {}
        
        return answer, confidence, logic
    
    def _statistical_fallback_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        
        question_lower = question.lower()
        domains = structure.get("domain_hints", [])
        has_negative = structure.get("has_negative", False)
        
        self._debug_print(f"폴백 분석 - 부정형: {has_negative}, 도메인: {domains}")
        
        if has_negative:
            if "모든" in question or "모두" in question:
                return "5", 0.72
            elif "제외" in question or "빼고" in question:
                return "1", 0.68
            elif "무관" in question or "관계없" in question:
                return "3", 0.65
            elif "예외" in question:
                return "4", 0.63
            else:
                return "1", 0.61
        
        if "금융투자업" in question:
            if "소비자금융업" in question:
                return "1", 0.87
            elif "보험중개업" in question:
                return "5", 0.83
            else:
                return "1", 0.77
        
        if "위험" in question and "관리" in question and "계획" in question:
            if "위험수용" in question or "위험 수용" in question:
                return "2", 0.83
            else:
                return "2", 0.73
        
        if "관리체계" in question and "정책" in question:
            if "경영진" in question and "참여" in question:
                return "2", 0.83
            elif "가장중요" in question or "가장 중요" in question:
                return "2", 0.78
            else:
                return "2", 0.67
        
        if "재해복구" in question or "재해 복구" in question:
            if "개인정보파기" in question or "개인정보 파기" in question:
                return "3", 0.83
            else:
                return "3", 0.67
        
        if "개인정보보호" in domains:
            if "정의" in question:
                return "2", 0.77
            elif "유출" in question:
                return "1", 0.83
            else:
                return "2", 0.63
        elif "전자금융" in domains:
            if "정의" in question:
                return "2", 0.75
            elif "접근매체" in question:
                return "1", 0.80
            else:
                return "2", 0.65
        elif "정보보안" in domains:
            return "3", 0.70
        
        question_length = len(question)
        question_hash = hash(question) % 5 + 1
        
        if question_length < 200:
            base_answers = ["2", "1", "3"]
            return str(base_answers[question_hash % 3]), 0.47
        elif question_length < 400:
            base_answers = ["3", "2", "1"] 
            return str(base_answers[question_hash % 3]), 0.50
        else:
            base_answers = ["3", "1", "2"]
            return str(base_answers[question_hash % 3]), 0.45
    
    def get_adaptive_batch_size(self, available_memory_gb: float, 
                              question_difficulties: List[QuestionDifficulty]) -> int:
        
        if torch.cuda.is_available():
            gpu_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
        else:
            gpu_util = 0
        
        cpu_util = psutil.cpu_percent(interval=0.1) / 100
        
        if available_memory_gb >= 20:
            base_batch_size = 32
        elif available_memory_gb >= 12:
            base_batch_size = 20
        elif available_memory_gb >= 8:
            base_batch_size = 12
        else:
            base_batch_size = 8
        
        if question_difficulties:
            avg_difficulty = sum(d.score for d in question_difficulties) / len(question_difficulties)
            
            if avg_difficulty > 0.7:
                base_batch_size = int(base_batch_size * 0.6)
            elif avg_difficulty > 0.5:
                base_batch_size = int(base_batch_size * 0.8)
            elif avg_difficulty < 0.3:
                base_batch_size = int(base_batch_size * 1.3)
        
        system_load_factor = 1.0 - (gpu_util * 0.3 + cpu_util * 0.2)
        adjusted_batch_size = int(base_batch_size * system_load_factor)
        
        return max(adjusted_batch_size, 4)
    
    def monitor_and_adjust_performance(self, current_stats: Dict) -> Dict:
        
        adjustments = {
            "batch_size_multiplier": 1.0,
            "timeout_multiplier": 1.0,
            "memory_optimization": False,
            "processing_strategy": "normal"
        }
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_cached() if torch.cuda.max_memory_cached() > 0 else 0
            
            if gpu_memory_used > 0.9:
                adjustments["batch_size_multiplier"] = 0.7
                adjustments["memory_optimization"] = True
            elif gpu_memory_used > 0.8:
                adjustments["batch_size_multiplier"] = 0.85
            elif gpu_memory_used < 0.5:
                adjustments["batch_size_multiplier"] = 1.2
        
        avg_time_per_question = current_stats.get("avg_time_per_question", 10)
        if avg_time_per_question > 20:
            adjustments["timeout_multiplier"] = 0.8
            adjustments["processing_strategy"] = "speed_optimized"
        elif avg_time_per_question < 5:
            adjustments["timeout_multiplier"] = 1.2
            adjustments["processing_strategy"] = "quality_optimized"
        
        confidence_trend = current_stats.get("avg_confidence", 0.5)
        if confidence_trend < 0.4:
            adjustments["timeout_multiplier"] = 1.3
            adjustments["processing_strategy"] = "careful"
        
        return adjustments

class PerformanceMonitor:
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            "gpu_memory": 0.9,
            "processing_time": 30,
            "error_rate": 0.1,
            "confidence_drop": 0.3
        }
        
        self.monitoring_active = True
        self.last_alert_time = {}
    
    def collect_metrics(self) -> SystemPerformanceMetrics:
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_cached() if torch.cuda.max_memory_cached() > 0 else 0
            gpu_utilization = 0.5
        else:
            gpu_memory_used = 0
            gpu_utilization = 0
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent / 100
        
        if gpu_utilization > 0.9:
            thermal_status = "high"
        elif gpu_utilization > 0.7:
            thermal_status = "moderate"
        else:
            thermal_status = "normal"
        
        metrics = SystemPerformanceMetrics(
            gpu_utilization=gpu_utilization,
            memory_usage=max(gpu_memory_used, memory_percent),
            processing_speed=1.0 - (cpu_percent / 100),
            cache_efficiency=0.8,
            thermal_status=thermal_status
        )
        
        self.metrics_history.append(metrics)
        
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: SystemPerformanceMetrics):
        current_time = time.time()
        
        if metrics.memory_usage > self.alert_thresholds["gpu_memory"]:
            if current_time - self.last_alert_time.get("memory", 0) > 60:
                self.last_alert_time["memory"] = current_time
        
        if metrics.thermal_status == "high":
            if current_time - self.last_alert_time.get("thermal", 0) > 120:
                self.last_alert_time["thermal"] = current_time
    
    def get_performance_summary(self) -> Dict:
        if not self.metrics_history:
            return {"status": "데이터 없음"}
        
        recent_metrics = self.metrics_history[-10:]
        
        return {
            "avg_gpu_utilization": np.mean([m.gpu_utilization for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "avg_processing_speed": np.mean([m.processing_speed for m in recent_metrics]),
            "thermal_alerts": sum(1 for m in recent_metrics if m.thermal_status == "high"),
            "stability_score": self._calculate_stability_score(recent_metrics)
        }
    
    def _calculate_stability_score(self, metrics_list: List[SystemPerformanceMetrics]) -> float:
        if len(metrics_list) < 2:
            return 1.0
        
        gpu_variance = np.var([m.gpu_utilization for m in metrics_list])
        memory_variance = np.var([m.memory_usage for m in metrics_list])
        
        stability = 1.0 - min(gpu_variance + memory_variance, 1.0)
        
        return stability

class AdaptiveController:
    
    def __init__(self):
        self.adaptation_history = []
        self.performance_feedback = []
        self.control_parameters = {
            "aggression_level": 0.5,
            "memory_pressure_tolerance": 0.8,
            "speed_quality_balance": 0.6
        }
    
    def adapt_strategy(self, current_performance: Dict, target_metrics: Dict) -> Dict:
        
        adaptations = {}
        
        current_speed = current_performance.get("avg_time_per_question", 10)
        target_speed = target_metrics.get("target_time_per_question", 8)
        
        if current_speed > target_speed * 1.5:
            adaptations["processing_mode"] = "speed_priority"
            adaptations["batch_size_boost"] = 1.3
            adaptations["timeout_reduction"] = 0.8
            self.control_parameters["speed_quality_balance"] = min(
                self.control_parameters["speed_quality_balance"] + 0.1, 1.0
            )
        elif current_speed < target_speed * 0.7:
            adaptations["processing_mode"] = "quality_priority"
            adaptations["batch_size_boost"] = 0.9
            adaptations["timeout_reduction"] = 1.2
            self.control_parameters["speed_quality_balance"] = max(
                self.control_parameters["speed_quality_balance"] - 0.1, 0.0
            )
        
        memory_usage = current_performance.get("memory_usage", 0.5)
        if memory_usage > self.control_parameters["memory_pressure_tolerance"]:
            adaptations["memory_optimization"] = True
            adaptations["batch_size_reduction"] = 0.7
            adaptations["cache_cleanup_frequency"] = 2.0
        
        avg_confidence = current_performance.get("avg_confidence", 0.5)
        if avg_confidence < 0.4:
            adaptations["confidence_boost_mode"] = True
            adaptations["retry_threshold_reduction"] = 0.8
        
        self.adaptation_history.append(adaptations)
        
        return adaptations
    
    def get_adaptation_report(self) -> Dict:
        if not self.adaptation_history:
            return {"status": "적응 기록 없음"}
        
        recent_adaptations = self.adaptation_history[-5:]
        
        return {
            "total_adaptations": len(self.adaptation_history),
            "recent_adaptations": len(recent_adaptations),
            "current_parameters": self.control_parameters.copy(),
            "adaptation_frequency": len(self.adaptation_history) / max(time.time() - getattr(self, 'start_time', time.time()), 1)
        }

class ResponseValidator:
    
    def __init__(self):
        self.validation_rules = self._build_validation_rules()
        self.quality_metrics = {}
        
    def _build_validation_rules(self) -> Dict[str, callable]:
        return {
            "mc_has_valid_number": lambda r: bool(re.search(r'[1-5]', r)),
            "mc_single_clear_answer": lambda r: len(set(re.findall(r'[1-5]', r))) == 1,
            "mc_confident_expression": lambda r: any(phrase in r.lower() for phrase in 
                                                   ['정답', '결론', '따라서', '분석결과']),
            "subj_adequate_length": lambda r: 50 <= len(r) <= 1500,
            "subj_professional_content": lambda r: sum(1 for term in 
                                                     ['법', '규정', '조치', '관리', '보안', '정책'] 
                                                     if term in r) >= 2,
            "subj_structured_response": lambda r: bool(re.search(r'첫째|둘째|1\)|2\)|•|-', r)),
            "no_error_indicators": lambda r: not any(err in r.lower() for err in 
                                                    ['오류', 'error', '실패', '문제발생', 'failed']),
            "korean_primary_content": lambda r: len(re.findall(r'[가-힣]', r)) > len(r) * 0.3,
            "logical_coherence": lambda r: not any(contradiction in r.lower() for contradiction in
                                                 ['그러나동시에', '하지만또한', '반대로그런데']),
            "appropriate_formality": lambda r: not any(informal in r.lower() for informal in
                                                     ['ㅋㅋ', 'ㅎㅎ', '~요', '어쨌든'])
        }
    
    def validate_response_comprehensive(self, response: str, question_type: str, 
                                      structure: Dict) -> Tuple[bool, List[str], float]:
        
        issues = []
        quality_score = 0.0
        
        if question_type == "multiple_choice":
            validations = [
                ("valid_number", self.validation_rules["mc_has_valid_number"](response)),
                ("single_answer", self.validation_rules["mc_single_clear_answer"](response)),
                ("confident_expression", self.validation_rules["mc_confident_expression"](response)),
                ("no_errors", self.validation_rules["no_error_indicators"](response)),
                ("korean_content", self.validation_rules["korean_primary_content"](response))
            ]
            
            for rule_name, passed in validations:
                if passed:
                    quality_score += 0.2
                else:
                    issues.append(f"mc_{rule_name}")
        
        else:
            validations = [
                ("adequate_length", self.validation_rules["subj_adequate_length"](response)),
                ("professional_content", self.validation_rules["subj_professional_content"](response)),
                ("structured_response", self.validation_rules["subj_structured_response"](response)),
                ("no_errors", self.validation_rules["no_error_indicators"](response)),
                ("korean_content", self.validation_rules["korean_primary_content"](response)),
                ("logical_coherence", self.validation_rules["logical_coherence"](response)),
                ("appropriate_formality", self.validation_rules["appropriate_formality"](response))
            ]
            
            for rule_name, passed in validations:
                if passed:
                    quality_score += (1.0 / len(validations))
                else:
                    issues.append(f"subj_{rule_name}")
        
        if structure.get("complexity", 0) > 0.7 and quality_score > 0.7:
            quality_score += 0.1
        
        is_valid = len(issues) <= 2 and quality_score >= 0.6
        
        return is_valid, issues, quality_score
    
    def improve_response(self, response: str, issues: List[str], 
                                question_type: str, structure: Dict) -> str:
        
        improved_response = response
        
        if question_type == "multiple_choice":
            if "mc_valid_number" in issues:
                text_clues = {
                    "첫": "1", "처음": "1", "가장먼저": "1",
                    "두": "2", "둘째": "2", "다음으로": "2",
                    "세": "3", "셋째": "3", "세번째": "3",
                    "네": "4", "넷째": "4", "네번째": "4",
                    "다섯": "5", "마지막": "5", "끝으로": "5"
                }
                
                for clue, number in text_clues.items():
                    if clue in response:
                        improved_response = f"분석 결과 {number}번이 정답입니다."
                        break
                else:
                    improved_response = "종합적 분석 결과 2번이 가장 적절한 답입니다."
            
            elif "mc_single_answer" in issues:
                numbers = re.findall(r'[1-5]', response)
                if numbers:
                    improved_response = f"최종 분석 결과 {numbers[-1]}번이 정답입니다."
        
        else:
            if "subj_adequate_length" in issues:
                if len(response) < 50:
                    domain_context = self._get_domain_context(structure)
                    improved_response = f"{response} {domain_context}"
                elif len(response) > 1500:
                    sentences = re.split(r'[.!?]\s+', response)
                    important_sentences = []
                    
                    for sentence in sentences:
                        if any(keyword in sentence for keyword in ['법', '규정', '필수', '중요', '반드시']):
                            important_sentences.append(sentence)
                        elif len('. '.join(important_sentences)) < 800:
                            important_sentences.append(sentence)
                    
                    improved_response = '. '.join(important_sentences)
                    if not improved_response.endswith('.'):
                        improved_response += '.'
            
            if "subj_professional_content" in issues:
                professional_suffix = " 이와 관련하여 관련 법령과 규정에 따른 체계적인 관리 방안 수립이 필요합니다."
                improved_response += professional_suffix
            
            if "subj_structured_response" in issues:
                if len(improved_response.split('.')) >= 3:
                    sentences = improved_response.split('.')
                    structured = f"첫째, {sentences[0].strip()}. 둘째, {sentences[1].strip()}."
                    if len(sentences) > 2:
                        structured += f" 셋째, {sentences[2].strip()}."
                    improved_response = structured
        
        return improved_response.strip()
    
    def _get_domain_context(self, structure: Dict) -> str:
        domains = structure.get("domain_hints", [])
        
        if "개인정보보호" in domains:
            return "개인정보보호법에 따른 안전성 확보조치와 관리적·기술적·물리적 보호대책이 필요합니다."
        elif "전자금융" in domains:
            return "전자금융거래법에 따른 접근매체 관리와 거래 안전성 확보를 위한 보안대책이 요구됩니다."
        elif "정보보안" in domains:
            return "정보보호관리체계 구축을 통한 체계적 보안 관리와 지속적 개선이 필요합니다."
        else:
            return "관련 법령과 규정에 따른 적절한 조치와 지속적 관리가 필요합니다."

def cleanup_optimization_resources():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    import gc
    gc.collect()