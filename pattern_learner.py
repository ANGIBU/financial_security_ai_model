# pattern_learner.py

import re
import json
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

def _default_int():
    return 0

def _default_float():
    return 0.0

def _default_list():
    return []

def _default_counter():
    return Counter()

class AnswerPatternLearner:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.patterns = {
            "keyword_answer_map": defaultdict(_default_counter),
            "domain_answer_distribution": defaultdict(_default_counter),
            "negative_answer_patterns": Counter(),
            "question_type_patterns": defaultdict(_default_counter),
            "context_answer_patterns": defaultdict(_default_counter),
            "structure_answer_patterns": defaultdict(_default_counter)
        }
        
        self.learned_rules = self._initialize_comprehensive_rules()
        
        self.pattern_performance = {
            "rule_success_rate": defaultdict(_default_list),
            "prediction_accuracy": defaultdict(_default_float),
            "confidence_tracking": defaultdict(_default_list)
        }
        
        self.prediction_cache = {}
        self.pattern_cache = {}
        self.max_cache_size = 500
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
        
    def _initialize_comprehensive_rules(self) -> Dict:
        return {
            "개인정보_정의": {
                "keywords": ["개인정보", "정의", "의미", "개념", "식별", "살아있는"],
                "preferred_answers": {"2": 0.75, "1": 0.18, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.88,
                "boost_keywords": ["살아있는", "개인", "알아볼", "식별할"]
            },
            "전자금융_정의": {
                "keywords": ["전자금융", "정의", "전자적", "거래", "장치"],
                "preferred_answers": {"2": 0.72, "3": 0.18, "1": 0.07, "4": 0.02, "5": 0.01},
                "confidence": 0.85,
                "boost_keywords": ["금융상품", "서비스", "제공"]
            },
            "유출_신고": {
                "keywords": ["유출", "신고", "즉시", "지체", "통지", "개인정보"],
                "preferred_answers": {"1": 0.78, "2": 0.15, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.90,
                "boost_keywords": ["지체없이", "정보주체"]
            },
            "금융투자업_분류": {
                "keywords": ["금융투자업", "구분", "해당하지", "소비자금융", "보험중개"],
                "preferred_answers": {"1": 0.82, "5": 0.12, "2": 0.04, "3": 0.01, "4": 0.01},
                "confidence": 0.95,
                "boost_keywords": ["소비자금융업", "보험중개업"]
            },
            "위험관리_계획": {
                "keywords": ["위험", "관리", "계획", "수립", "고려", "요소"],
                "preferred_answers": {"2": 0.80, "1": 0.12, "3": 0.05, "4": 0.02, "5": 0.01},
                "confidence": 0.88,
                "boost_keywords": ["위험수용", "대응전략"]
            },
            "관리체계_정책": {
                "keywords": ["관리체계", "정책", "수립", "단계", "중요"],
                "preferred_answers": {"2": 0.78, "1": 0.15, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.87,
                "boost_keywords": ["경영진", "참여", "지원"]
            },
            "재해복구_계획": {
                "keywords": ["재해", "복구", "계획", "수립", "고려"],
                "preferred_answers": {"3": 0.80, "1": 0.10, "2": 0.06, "4": 0.02, "5": 0.02},
                "confidence": 0.90,
                "boost_keywords": ["개인정보파기", "파기절차"]
            },
            "접근매체_관리": {
                "keywords": ["접근매체", "선정", "관리", "안전", "신뢰"],
                "preferred_answers": {"1": 0.75, "2": 0.18, "3": 0.05, "4": 0.01, "5": 0.01},
                "confidence": 0.85,
                "boost_keywords": ["금융회사", "안전하고"]
            },
            "안전성_확보": {
                "keywords": ["안전성", "확보조치", "기술적", "관리적", "물리적"],
                "preferred_answers": {"1": 0.70, "2": 0.22, "3": 0.06, "4": 0.01, "5": 0.01},
                "confidence": 0.83,
                "boost_keywords": ["보호대책", "필요한"]
            },
            "부정형_일반": {
                "keywords": ["해당하지", "적절하지", "옳지", "틀린", "잘못된"],
                "preferred_answers": {"1": 0.38, "3": 0.25, "5": 0.20, "2": 0.12, "4": 0.05},
                "confidence": 0.72,
                "boost_keywords": ["않는", "않은", "아닌"]
            },
            "모두_포함": {
                "keywords": ["모두", "모든", "전부", "다음중"],
                "preferred_answers": {"5": 0.50, "1": 0.25, "4": 0.15, "3": 0.07, "2": 0.03},
                "confidence": 0.78,
                "boost_keywords": ["해당하는", "포함되는"]
            },
            "ISMS_관련": {
                "keywords": ["ISMS", "정보보호", "관리체계", "인증"],
                "preferred_answers": {"3": 0.65, "2": 0.22, "1": 0.10, "4": 0.02, "5": 0.01},
                "confidence": 0.80,
                "boost_keywords": ["운영", "구축"]
            },
            "암호화_요구": {
                "keywords": ["암호화", "암호", "복호화", "키관리"],
                "preferred_answers": {"2": 0.62, "1": 0.25, "3": 0.10, "4": 0.02, "5": 0.01},
                "confidence": 0.78,
                "boost_keywords": ["대칭키", "공개키", "해시"]
            },
            "전자서명_법령": {
                "keywords": ["전자서명", "전자서명법", "인증", "공개키"],
                "preferred_answers": {"2": 0.68, "1": 0.20, "3": 0.08, "4": 0.03, "5": 0.01},
                "confidence": 0.80,
                "boost_keywords": ["전자서명법", "공인인증"]
            },
            "신용정보_보호": {
                "keywords": ["신용정보", "신용정보법", "보호", "이용"],
                "preferred_answers": {"1": 0.70, "2": 0.18, "3": 0.08, "4": 0.03, "5": 0.01},
                "confidence": 0.82,
                "boost_keywords": ["신용정보보호법", "동의"]
            },
            "금융실명_거래": {
                "keywords": ["금융실명", "실명거래", "비실명", "예외"],
                "preferred_answers": {"2": 0.65, "3": 0.20, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["금융실명법", "비실명거래"]
            },
            "보험업법_관련": {
                "keywords": ["보험업법", "보험", "모집", "설계사"],
                "preferred_answers": {"3": 0.60, "2": 0.25, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["보험설계사", "모집행위"]
            },
            "자본시장법_관련": {
                "keywords": ["자본시장법", "자본시장", "금융투자", "투자자"],
                "preferred_answers": {"2": 0.58, "1": 0.25, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["투자자보호", "자본시장"]
            },
            "은행법_관련": {
                "keywords": ["은행법", "은행", "예금", "대출"],
                "preferred_answers": {"1": 0.62, "2": 0.23, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.76,
                "boost_keywords": ["은행업무", "예금보험"]
            },
            "IT_거버넌스": {
                "keywords": ["IT거버넌스", "IT", "거버넌스", "정보기술"],
                "preferred_answers": {"3": 0.55, "2": 0.28, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["IT전략", "정보기술"]
            },
            "COBIT_관련": {
                "keywords": ["COBIT", "IT관리", "프레임워크"],
                "preferred_answers": {"2": 0.60, "3": 0.25, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["IT거버넌스", "관리"]
            },
            "ITIL_관련": {
                "keywords": ["ITIL", "서비스", "IT서비스"],
                "preferred_answers": {"3": 0.58, "2": 0.25, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.74,
                "boost_keywords": ["서비스관리", "IT서비스"]
            },
            "ISO27001_관련": {
                "keywords": ["ISO27001", "ISO", "27001", "정보보호"],
                "preferred_answers": {"3": 0.62, "2": 0.23, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["정보보호관리", "인증"]
            },
            "PCI_DSS": {
                "keywords": ["PCI", "DSS", "카드", "결제"],
                "preferred_answers": {"2": 0.58, "1": 0.25, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["결제카드", "보안표준"]
            },
            "SOX_법령": {
                "keywords": ["SOX", "사베인스", "내부통제"],
                "preferred_answers": {"2": 0.55, "3": 0.25, "1": 0.15, "4": 0.03, "5": 0.02},
                "confidence": 0.70,
                "boost_keywords": ["내부통제", "재무보고"]
            },
            "바젤_협약": {
                "keywords": ["바젤", "basel", "자본", "적정성"],
                "preferred_answers": {"1": 0.60, "2": 0.25, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["자본적정성", "Basel"]
            },
            "GDPR_관련": {
                "keywords": ["GDPR", "개인정보", "유럽", "EU"],
                "preferred_answers": {"2": 0.62, "1": 0.23, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["개인정보보호", "유럽연합"]
            },
            "CCPA_관련": {
                "keywords": ["CCPA", "캘리포니아", "소비자", "개인정보"],
                "preferred_answers": {"2": 0.58, "1": 0.25, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["소비자개인정보", "캘리포니아"]
            },
            "해킹_공격": {
                "keywords": ["해킹", "공격", "침입", "사이버"],
                "preferred_answers": {"3": 0.55, "1": 0.25, "2": 0.15, "4": 0.03, "5": 0.02},
                "confidence": 0.70,
                "boost_keywords": ["사이버공격", "침해"]
            },
            "악성코드_분류": {
                "keywords": ["악성코드", "malware", "바이러스", "웜"],
                "preferred_answers": {"2": 0.58, "3": 0.25, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["트로이", "랜섬웨어"]
            },
            "트로이목마_특징": {
                "keywords": ["트로이", "trojan", "원격", "제어"],
                "preferred_answers": {"2": 0.62, "1": 0.23, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.80,
                "boost_keywords": ["원격제어", "원격접근"]
            },
            "랜섬웨어_대응": {
                "keywords": ["랜섬웨어", "ransomware", "암호화", "복구"],
                "preferred_answers": {"1": 0.65, "2": 0.20, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["백업", "복구"]
            },
            "피싱_공격": {
                "keywords": ["피싱", "phishing", "사기", "이메일"],
                "preferred_answers": {"3": 0.60, "2": 0.23, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["스피어피싱", "사회공학"]
            },
            "스미싱_공격": {
                "keywords": ["스미싱", "smishing", "SMS", "문자"],
                "preferred_answers": {"3": 0.58, "2": 0.25, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["문자메시지", "SMS"]
            },
            "파밍_공격": {
                "keywords": ["파밍", "pharming", "DNS", "도메인"],
                "preferred_answers": {"2": 0.62, "3": 0.23, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["DNS변조", "도메인"]
            },
            "DDoS_공격": {
                "keywords": ["DDoS", "분산", "서비스", "거부"],
                "preferred_answers": {"1": 0.58, "2": 0.25, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.73,
                "boost_keywords": ["분산서비스거부", "트래픽"]
            },
            "APT_공격": {
                "keywords": ["APT", "지능형", "지속적", "위협"],
                "preferred_answers": {"2": 0.60, "3": 0.23, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.76,
                "boost_keywords": ["지능형지속위협", "표적"]
            },
            "제로데이_공격": {
                "keywords": ["제로데이", "zero-day", "취약점"],
                "preferred_answers": {"3": 0.62, "2": 0.23, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["미패치", "취약점"]
            },
            "백도어_설치": {
                "keywords": ["백도어", "backdoor", "은밀", "접근"],
                "preferred_answers": {"2": 0.58, "1": 0.25, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["은밀한", "우회"]
            },
            "루트킷_은닉": {
                "keywords": ["루트킷", "rootkit", "은닉", "탐지"],
                "preferred_answers": {"3": 0.60, "2": 0.23, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["시스템은닉", "탐지회피"]
            },
            "키로거_정보수집": {
                "keywords": ["키로거", "keylogger", "키보드", "입력"],
                "preferred_answers": {"2": 0.62, "1": 0.23, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["키보드입력", "정보수집"]
            },
            "스파이웨어_감시": {
                "keywords": ["스파이웨어", "spyware", "감시", "정보"],
                "preferred_answers": {"2": 0.58, "3": 0.25, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["정보수집", "사용자감시"]
            },
            "애드웨어_광고": {
                "keywords": ["애드웨어", "adware", "광고", "팝업"],
                "preferred_answers": {"3": 0.55, "2": 0.28, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.68,
                "boost_keywords": ["광고표시", "팝업"]
            },
            "방화벽_정책": {
                "keywords": ["방화벽", "firewall", "정책", "규칙"],
                "preferred_answers": {"1": 0.62, "2": 0.23, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["접근제어", "네트워크"]
            },
            "IDS_IPS": {
                "keywords": ["IDS", "IPS", "침입", "탐지"],
                "preferred_answers": {"2": 0.60, "3": 0.23, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.76,
                "boost_keywords": ["침입탐지", "침입방지"]
            },
            "백업_복구": {
                "keywords": ["백업", "backup", "복구", "recovery"],
                "preferred_answers": {"1": 0.65, "2": 0.20, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.80,
                "boost_keywords": ["데이터복구", "백업전략"]
            },
            "비즈니스연속성": {
                "keywords": ["비즈니스", "연속성", "BCP", "업무"],
                "preferred_answers": {"2": 0.58, "1": 0.25, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["업무연속성", "BCP"]
            },
            "접근제어_모델": {
                "keywords": ["접근제어", "access", "control", "권한"],
                "preferred_answers": {"2": 0.62, "3": 0.23, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["권한관리", "인증"]
            },
            "다중인증_요소": {
                "keywords": ["다중인증", "MFA", "2FA", "이중"],
                "preferred_answers": {"1": 0.60, "2": 0.25, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.77,
                "boost_keywords": ["이중인증", "다요소"]
            },
            "생체인증_방식": {
                "keywords": ["생체인증", "지문", "홍채", "얼굴"],
                "preferred_answers": {"3": 0.58, "2": 0.25, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.74,
                "boost_keywords": ["바이오메트릭", "생체정보"]
            },
            "취약점_평가": {
                "keywords": ["취약점", "vulnerability", "평가", "점검"],
                "preferred_answers": {"2": 0.60, "1": 0.23, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.76,
                "boost_keywords": ["보안점검", "취약성"]
            },
            "모의해킹_테스트": {
                "keywords": ["모의해킹", "penetration", "testing", "침투"],
                "preferred_answers": {"3": 0.62, "2": 0.23, "1": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["침투테스트", "모의침투"]
            },
            "보안교육_훈련": {
                "keywords": ["보안교육", "훈련", "인식", "교육"],
                "preferred_answers": {"2": 0.58, "1": 0.25, "3": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["보안인식", "사용자교육"]
            },
            "암호정책_관리": {
                "keywords": ["암호정책", "password", "policy", "복잡성"],
                "preferred_answers": {"1": 0.62, "2": 0.23, "3": 0.10, "4": 0.03, "5": 0.02},
                "confidence": 0.78,
                "boost_keywords": ["패스워드정책", "복잡성"]
            },
            "소셜엔지니어링": {
                "keywords": ["소셜", "엔지니어링", "사회공학", "심리"],
                "preferred_answers": {"3": 0.60, "2": 0.23, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.75,
                "boost_keywords": ["사회공학", "인간심리"]
            },
            "클라우드_보안": {
                "keywords": ["클라우드", "cloud", "보안", "SaaS"],
                "preferred_answers": {"2": 0.58, "3": 0.25, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.72,
                "boost_keywords": ["클라우드보안", "가상화"]
            },
            "IoT_보안": {
                "keywords": ["IoT", "사물인터넷", "디바이스", "연결"],
                "preferred_answers": {"3": 0.58, "2": 0.25, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.70,
                "boost_keywords": ["사물인터넷", "스마트디바이스"]
            },
            "모바일_보안": {
                "keywords": ["모바일", "mobile", "스마트폰", "앱"],
                "preferred_answers": {"2": 0.60, "3": 0.23, "1": 0.12, "4": 0.03, "5": 0.02},
                "confidence": 0.73,
                "boost_keywords": ["모바일보안", "앱보안"]
            }
        }
    
    def analyze_question_pattern(self, question: str) -> Optional[Dict]:
        q_hash = hash(question[:100])
        if q_hash in self.pattern_cache:
            return self.pattern_cache[q_hash]
        
        question_lower = question.lower().replace(" ", "")
        
        best_rule = None
        best_score = 0
        
        for rule_name, rule_info in self.learned_rules.items():
            keywords = rule_info["keywords"]
            boost_keywords = rule_info.get("boost_keywords", [])
            
            base_matches = sum(1 for kw in keywords if kw.replace(" ", "") in question_lower)
            
            if base_matches > 0:
                base_score = base_matches / len(keywords)
                
                boost_score = 0
                for boost_kw in boost_keywords:
                    if boost_kw.replace(" ", "") in question_lower:
                        boost_score += 0.18
                
                final_score = base_score * (1 + boost_score)
                
                if final_score > best_score:
                    best_score = final_score
                    best_rule = {
                        "rule": rule_name,
                        "match_score": final_score,
                        "base_confidence": rule_info["confidence"],
                        "answers": rule_info["preferred_answers"]
                    }
        
        if len(self.pattern_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.pattern_cache))
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[q_hash] = best_rule
        return best_rule
    
    def predict_answer(self, question: str, structure: Dict) -> Tuple[str, float]:
        cache_key = hash(f"{question[:50]}{structure.get('question_type', '')}")
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        if structure.get("has_negative", False):
            result = self._predict_negative_enhanced(question, structure)
        elif structure.get("has_all_option", False):
            result = self._predict_all_option(question, structure)
        else:
            pattern_match = self.analyze_question_pattern(question)
            
            if pattern_match and pattern_match["base_confidence"] > 0.60:
                answers = pattern_match["answers"]
                best_answer = max(answers.items(), key=lambda x: x[1])
                confidence = pattern_match["base_confidence"] * pattern_match["match_score"]
                result = (best_answer[0], min(confidence, 0.95))
            else:
                result = self._statistical_prediction_enhanced(question, structure)
        
        if len(self.prediction_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[cache_key] = result
        return result
    
    def _predict_negative_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        question_lower = question.lower()
        
        if "모든" in question_lower or "모두" in question_lower:
            if "해당하지" in question_lower:
                return "5", 0.85
            else:
                return "1", 0.82
        elif "제외" in question_lower or "빼고" in question_lower:
            return "1", 0.78
        elif "예외" in question_lower:
            return "4", 0.72
        elif "무관" in question_lower or "관계없" in question_lower:
            return "3", 0.70
        else:
            domains = structure.get("domain_hints", [])
            if "개인정보보호" in domains:
                return "1", 0.68
            elif "전자금융" in domains:
                return "2", 0.68
            elif "정보보안" in domains:
                return "3", 0.65
            else:
                return "1", 0.62
    
    def _predict_all_option(self, question: str, structure: Dict) -> Tuple[str, float]:
        choices = structure.get("choices", [])
        if choices:
            last_choice_num = choices[-1].get("number", "5")
            return last_choice_num, 0.75
        return "5", 0.70
    
    def _statistical_prediction_enhanced(self, question: str, structure: Dict) -> Tuple[str, float]:
        if structure["question_type"] != "multiple_choice":
            return "", 0.15
        
        domains = structure.get("domain_hints", [])
        length = len(question)
        
        if "개인정보보호" in domains:
            if "정의" in question:
                return "2", 0.75
            elif "유출" in question:
                return "1", 0.78
            else:
                return "2", 0.58
        elif "전자금융" in domains:
            if "정의" in question:
                return "2", 0.72
            elif "접근매체" in question:
                return "1", 0.75
            else:
                return "2", 0.55
        elif "정보보안" in domains or "ISMS" in question:
            return "3", 0.62
        elif "사이버보안" in domains:
            if "트로이" in question:
                return "2", 0.78
            elif "악성코드" in question:
                return "2", 0.72
            else:
                return "3", 0.60
        
        if length < 200:
            return "2", 0.48
        elif length < 400:
            return "3", 0.45
        else:
            return "3", 0.42
    
    def update_patterns(self, question: str, correct_answer: str, structure: Dict,
                       prediction_result: Optional[Tuple[str, float]] = None):
        
        keywords = re.findall(r'[가-힣]{2,}', question.lower())
        for keyword in keywords[:5]:
            self.patterns["keyword_answer_map"][keyword][correct_answer] += 1
        
        domains = structure.get("domain_hints", ["일반"])
        for domain in domains:
            self.patterns["domain_answer_distribution"][domain][correct_answer] += 1
        
        if structure.get("has_negative", False):
            self.patterns["negative_answer_patterns"][correct_answer] += 1
        
        if structure.get("has_all_option", False):
            self.patterns["structure_answer_patterns"]["all_option"][correct_answer] += 1
        
        question_length = len(question)
        if question_length < 200:
            self.patterns["structure_answer_patterns"]["short"][correct_answer] += 1
        elif question_length < 400:
            self.patterns["structure_answer_patterns"]["medium"][correct_answer] += 1
        else:
            self.patterns["structure_answer_patterns"]["long"][correct_answer] += 1
        
        if prediction_result:
            predicted_answer, confidence = prediction_result
            is_correct = (predicted_answer == correct_answer)
            
            pattern_match = self.analyze_question_pattern(question)
            if pattern_match:
                rule_name = pattern_match["rule"]
                self.pattern_performance["rule_success_rate"][rule_name].append(is_correct)
                self.pattern_performance["confidence_tracking"][rule_name].append(confidence)
    
    def get_confidence_boost(self, question: str, predicted_answer: str, structure: Dict) -> float:
        boost = 0.0
        
        pattern_match = self.analyze_question_pattern(question)
        
        if pattern_match:
            answers = pattern_match["answers"]
            if predicted_answer in answers:
                preference_score = answers[predicted_answer]
                boost += preference_score * 0.15
        
        domains = structure.get("domain_hints", [])
        if domains and len(domains) == 1:
            boost += 0.10
        elif domains and len(domains) == 2:
            boost += 0.06
        
        if structure.get("has_negative", False) and predicted_answer in ["1", "5"]:
            boost += 0.08
        
        if structure.get("has_all_option", False) and predicted_answer == "5":
            boost += 0.10
        
        return min(boost, 0.30)
    
    def get_pattern_insights(self) -> Dict:
        insights = {
            "rule_performance": {},
            "domain_preferences": {},
            "negative_distribution": dict(self.patterns["negative_answer_patterns"]),
            "structure_patterns": {}
        }
        
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 3:
                success_rate = sum(success_list) / len(success_list)
                confidence_list = self.pattern_performance["confidence_tracking"][rule_name]
                avg_confidence = sum(confidence_list) / len(confidence_list) if confidence_list else 0
                
                insights["rule_performance"][rule_name] = {
                    "success_rate": success_rate,
                    "sample_size": len(success_list),
                    "avg_confidence": avg_confidence
                }
        
        for domain, answer_dist in self.patterns["domain_answer_distribution"].items():
            if sum(answer_dist.values()) >= 3:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["domain_preferences"][domain] = preferences
        
        for structure_type, answer_dist in self.patterns["structure_answer_patterns"].items():
            if isinstance(answer_dist, Counter) and sum(answer_dist.values()) >= 2:
                total = sum(answer_dist.values())
                preferences = {ans: count/total for ans, count in answer_dist.items()}
                insights["structure_patterns"][structure_type] = preferences
        
        return insights
    
    def optimize_rules(self):
        for rule_name, success_list in self.pattern_performance["rule_success_rate"].items():
            if len(success_list) >= 8:
                success_rate = sum(success_list) / len(success_list)
                
                if success_rate < 0.30 and rule_name in self.learned_rules:
                    self.learned_rules[rule_name]["confidence"] *= 0.90
                elif success_rate > 0.80 and rule_name in self.learned_rules:
                    current_confidence = self.learned_rules[rule_name]["confidence"]
                    self.learned_rules[rule_name]["confidence"] = min(current_confidence * 1.08, 0.98)
        
        for rule_name in list(self.learned_rules.keys()):
            if rule_name in self.pattern_performance["rule_success_rate"]:
                success_list = self.pattern_performance["rule_success_rate"][rule_name]
                if len(success_list) > 30:
                    self.pattern_performance["rule_success_rate"][rule_name] = success_list[-30:]
    
    def save_patterns(self, filepath: str = "./learned_patterns.pkl"):
        save_data = {
            "patterns": {k: dict(v) if hasattr(v, 'items') else v for k, v in self.patterns.items()},
            "rules": self.learned_rules,
            "performance": {k: dict(v) if hasattr(v, 'items') else v for k, v in self.pattern_performance.items()}
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def load_patterns(self, filepath: str = "./learned_patterns.pkl"):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                self.patterns = defaultdict(_default_counter, data.get("patterns", {}))
                self.learned_rules = data.get("rules", self.learned_rules)
                
                if "performance" in data:
                    perf_data = data["performance"]
                    self.pattern_performance = defaultdict(_default_list, perf_data)
                
                return True
        except Exception:
            return False
    
    def cleanup(self):
        cache_size = len(self.prediction_cache) + len(self.pattern_cache)
        if cache_size > 0 and self.debug_mode:
            print(f"패턴 학습기 캐시: {cache_size}개")
        
        self.prediction_cache.clear()
        self.pattern_cache.clear()

class SmartAnswerSelector:
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.pattern_learner = AnswerPatternLearner(debug_mode=debug_mode)
        self.selection_stats = {
            "total_selections": 0,
            "pattern_based": 0,
            "model_based": 0,
            "high_confidence": 0
        }
        
    def _debug_print(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def select_best_answer(self, question: str, model_response: str, 
                         structure: Dict, confidence: float) -> Tuple[str, float]:
        
        self.selection_stats["total_selections"] += 1
        
        extracted_answers = self._extract_answers_enhanced(model_response)
        
        if extracted_answers:
            answer = extracted_answers[0]
            
            confidence_boost = self.pattern_learner.get_confidence_boost(question, answer, structure)
            final_confidence = min(confidence + confidence_boost, 0.98)
            
            if final_confidence > 0.70:
                self.selection_stats["high_confidence"] += 1
            
            self.selection_stats["model_based"] += 1
            return answer, final_confidence
        
        pattern_answer, pattern_conf = self.pattern_learner.predict_answer(question, structure)
        
        if pattern_conf > 0.60:
            self.selection_stats["high_confidence"] += 1
        
        self.selection_stats["pattern_based"] += 1
        return pattern_answer, pattern_conf
    
    def _extract_answers_enhanced(self, response: str) -> List[str]:
        priority_patterns = [
            r'정답[:\s]*([1-5])',
            r'최종\s*답[:\s]*([1-5])',
            r'답[:\s]*([1-5])',
            r'([1-5])번이\s*정답',
            r'([1-5])번'
        ]
        
        for pattern in priority_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches
        
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            number_counts = {}
            for num in numbers:
                number_counts[num] = number_counts.get(num, 0) + 1
            
            if number_counts:
                sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
                return [num for num, _ in sorted_numbers]
        
        return []
    
    def get_selection_report(self) -> Dict:
        total = self.selection_stats["total_selections"]
        
        if total == 0:
            return {"message": "기록 없음"}
        
        return {
            "total_selections": total,
            "model_based_rate": self.selection_stats["model_based"] / total,
            "pattern_based_rate": self.selection_stats["pattern_based"] / total,
            "high_confidence_rate": self.selection_stats["high_confidence"] / total
        }
    
    def cleanup(self):
        total = self.selection_stats["total_selections"]
        if total > 0 and self.debug_mode:
            print(f"답변 선택기: {total}회 선택")
        
        self.pattern_learner.cleanup()