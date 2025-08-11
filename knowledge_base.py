# knowledge_base.py

"""
금융보안 지식베이스
- 도메인별 키워드 관리
- 문제 분석 및 분류
- 법령 참조 정보
- 핵심 개념 정의
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class FinancialSecurityKnowledgeBase:
    
    def __init__(self):
        self.domain_keywords = self._initialize_domain_keywords()
        self.law_references = self._initialize_law_references()
        self.security_concepts = self._initialize_security_concepts()
        self.technical_concepts = self._initialize_technical_concepts()
        
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        return {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보처리자", "개인정보보호법", "개인정보처리",
                "수집", "이용", "제공", "파기", "동의", "열람", "정정", "삭제",
                "민감정보", "고유식별정보", "안전성확보조치", "영향평가"
            ],
            "전자금융": [
                "전자금융거래", "전자금융거래법", "전자적장치", "접근매체", "전자서명",
                "전자인증", "금융기관", "전자금융업", "전자지급수단", "전자화폐",
                "오류정정", "손해배상", "약관", "이용자"
            ],
            "정보보안": [
                "정보보안", "정보보호", "정보보안관리체계", "ISMS", "ISMS-P",
                "보안정책", "보안통제", "위험관리", "취약점", "보안사고",
                "접근통제", "암호화", "네트워크보안", "시스템보안", "데이터보안"
            ],
            "사이버보안": [
                "사이버보안", "사이버공격", "해킹", "악성코드", "멀웨어", "바이러스",
                "웜", "트로이목마", "랜섬웨어", "스파이웨어", "애드웨어", "루트킷",
                "피싱", "스미싱", "파밍", "스피어피싱", "사회공학", "제로데이",
                "APT", "DDoS", "봇넷", "C&C", "백도어", "키로거"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험분석", "위험식별", "위험측정", "위험통제",
                "위험모니터링", "위험보고", "위험수용", "위험회피", "위험전가", "위험완화",
                "위험관리체계", "위험관리정책", "위험관리조직", "위험관리절차"
            ],
            "관리체계": [
                "관리체계", "정보보호관리체계", "정보보안관리체계", "PDCA",
                "정책", "조직", "자산관리", "인적보안", "물리보안", "시스템보안",
                "네트워크보안", "접근통제", "시스템개발", "공급업체관리", "사고관리"
            ],
            "재해복구": [
                "재해복구", "재해복구계획", "BCP", "업무연속성", "재해복구센터", "DRP",
                "백업", "복구", "RTO", "RPO", "복구목표시간", "복구목표시점",
                "핫사이트", "콜드사이트", "웜사이트"
            ],
            "금융투자업": [
                "금융투자업", "투자매매업", "투자중개업", "투자자문업", "투자일임업",
                "집합투자업", "신탁업", "소비자금융업", "보험중개업", "자본시장법"
            ],
            "암호화": [
                "암호화", "복호화", "암호", "암호키", "대칭키", "공개키", "개인키",
                "해시함수", "전자서명", "인증서", "PKI", "암호알고리즘",
                "AES", "RSA", "SHA", "MD5", "키관리", "키분배"
            ]
        }
    
    def _initialize_law_references(self) -> Dict[str, Dict]:
        return {
            "개인정보보호법": {
                "정의": "개인정보보호법 제2조",
                "처리원칙": "개인정보보호법 제3조",
                "수집제한": "개인정보보호법 제15조",
                "이용제한": "개인정보보호법 제18조",
                "제공제한": "개인정보보호법 제17조",
                "파기": "개인정보보호법 제21조",
                "안전성확보조치": "개인정보보호법 제29조",
                "유출신고": "개인정보보호법 제34조"
            },
            "전자금융거래법": {
                "정의": "전자금융거래법 제2조",
                "접근매체": "전자금융거래법 제2조 제10호",
                "이용자보호": "전자금융거래법 제9조",
                "거래내역통지": "전자금융거래법 제18조",
                "오류정정": "전자금융거래법 제19조",
                "손해배상": "전자금융거래법 제20조"
            },
            "정보통신망법": {
                "개인정보보호": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
                "개인정보수집": "정보통신망법 제22조",
                "개인정보이용제공": "정보통신망법 제24조",
                "개인정보보호조치": "정보통신망법 제28조"
            }
        }
    
    def _initialize_security_concepts(self) -> Dict[str, str]:
        return {
            "기밀성": "인가되지 않은 개인, 개체, 프로세스에 대해 정보를 사용하지 못하게 하거나 공개하지 않는 특성",
            "무결성": "정확성과 완전성을 보호하는 특성",
            "가용성": "인가된 개체가 요구할 때 접근 및 사용이 가능한 특성",
            "부인방지": "어떤 행위나 사건의 발생에 대해 나중에 부인할 수 없도록 하는 특성",
            "인증": "어떤 개체의 신원이 주장된 신원과 같음을 확실히 하는 과정",
            "인가": "특정 자원에 대한 접근 권한을 부여하는 과정",
            "식별": "사용자나 프로세스가 자신을 시스템에 알리는 과정"
        }
    
    def _initialize_technical_concepts(self) -> Dict[str, str]:
        return {
            "트로이목마": "정상적인 프로그램으로 위장하여 시스템에 침입한 후 악의적인 행위를 수행하는 악성코드",
            "RAT": "원격 접근 트로이목마로, 공격자가 감염된 시스템을 원격으로 제어할 수 있게 하는 악성코드",
            "랜섬웨어": "시스템의 파일을 암호화하여 사용자가 접근할 수 없게 한 후 복구를 대가로 금전을 요구하는 악성코드",
            "피싱": "가짜 웹사이트나 이메일을 통해 개인정보나 금융정보를 탈취하는 공격",
            "스미싱": "SMS를 이용한 피싱 공격",
            "파밍": "DNS 변조를 통해 정상적인 도메인으로 접속해도 가짜 사이트로 연결되게 하는 공격",
            "DDoS": "분산 서비스 거부 공격으로, 다수의 시스템을 이용해 대상 시스템에 과부하를 일으키는 공격",
            "APT": "지능형 지속 위협으로, 특정 조직을 대상으로 장기간에 걸쳐 지속적이고 은밀하게 수행되는 공격",
            "제로데이": "보안 취약점이 발견되었지만 아직 패치가 제공되지 않은 상태에서 발생하는 공격",
            "소셜엔지니어링": "기술적 수단보다는 인간의 심리적 약점을 이용하여 정보를 획득하는 공격 기법"
        }
    
    def analyze_question(self, question: str) -> Dict:
        question_lower = question.lower()
        
        analysis = {
            "domain": [],
            "complexity": self._analyze_complexity(question),
            "question_type": self._determine_question_type(question),
            "key_concepts": [],
            "law_references": [],
            "difficulty_level": self._assess_difficulty(question)
        }
        
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            if matches > 0:
                analysis["domain"].append(domain)
        
        for concept, definition in self.security_concepts.items():
            if concept in question_lower:
                analysis["key_concepts"].append(concept)
        
        for law, sections in self.law_references.items():
            if any(section_key in question_lower for section_key in sections.keys()):
                analysis["law_references"].append(law)
        
        return analysis
    
    def _analyze_complexity(self, question: str) -> float:
        complexity_factors = {
            "length": min(len(question) / 1000, 0.3),
            "technical_terms": 0,
            "legal_terms": 0,
            "multiple_concepts": 0
        }
        
        question_lower = question.lower()
        
        technical_terms = ["암호화", "해시", "PKI", "SSL", "TLS", "VPN", "IDS", "IPS"]
        complexity_factors["technical_terms"] = min(
            sum(1 for term in technical_terms if term in question_lower) * 0.1, 0.2
        )
        
        legal_terms = ["법", "규정", "조항", "시행령", "고시"]
        complexity_factors["legal_terms"] = min(
            sum(1 for term in legal_terms if term in question_lower) * 0.05, 0.15
        )
        
        concept_count = sum(1 for domain_keywords in self.domain_keywords.values() 
                           for keyword in domain_keywords if keyword in question_lower)
        complexity_factors["multiple_concepts"] = min(concept_count * 0.03, 0.2)
        
        return sum(complexity_factors.values())
    
    def _determine_question_type(self, question: str) -> str:
        question_lower = question.lower()
        
        if any(indicator in question_lower for indicator in 
               ["설명하세요", "기술하세요", "서술하세요", "작성하세요"]):
            return "descriptive"
        elif any(indicator in question_lower for indicator in 
                 ["정의", "의미", "개념"]):
            return "definitional"
        elif any(indicator in question_lower for indicator in 
                 ["방안", "대책", "조치"]):
            return "solution_oriented"
        elif any(indicator in question_lower for indicator in 
                 ["다음 중", "가장 적절한", "옳은 것"]):
            return "multiple_choice"
        else:
            return "general"
    
    def _assess_difficulty(self, question: str) -> str:
        complexity = self._analyze_complexity(question)
        question_lower = question.lower()
        
        advanced_indicators = [
            "고급", "심화", "전문", "세부", "상세",
            "분석", "평가", "설계", "구현"
        ]
        
        basic_indicators = [
            "기본", "초급", "개념", "정의", "의미"
        ]
        
        if any(indicator in question_lower for indicator in advanced_indicators) or complexity > 0.6:
            return "advanced"
        elif any(indicator in question_lower for indicator in basic_indicators) or complexity < 0.3:
            return "basic"
        else:
            return "intermediate"
    
    def get_domain_context(self, domain: str) -> Dict:
        if domain not in self.domain_keywords:
            return {}
        
        context = {
            "keywords": self.domain_keywords[domain],
            "related_laws": [],
            "key_concepts": []
        }
        
        if domain == "개인정보보호":
            context["related_laws"] = ["개인정보보호법", "정보통신망법"]
            context["key_concepts"] = ["수집", "이용", "제공", "파기", "동의"]
        elif domain == "전자금융":
            context["related_laws"] = ["전자금융거래법"]
            context["key_concepts"] = ["접근매체", "전자서명", "이용자보호"]
        elif domain == "정보보안":
            context["related_laws"] = ["정보통신망법"]
            context["key_concepts"] = ["기밀성", "무결성", "가용성"]
        
        return context
    
    def suggest_answer_structure(self, question_type: str, domain: List[str]) -> List[str]:
        if question_type == "descriptive":
            return [
                "개념 정의",
                "주요 특징",
                "구체적 방법",
                "예시 또는 사례",
                "결론 및 요약"
            ]
        elif question_type == "definitional":
            return [
                "법적 정의",
                "핵심 요소",
                "적용 범위",
                "관련 개념과의 차이점"
            ]
        elif question_type == "solution_oriented":
            return [
                "현황 분석",
                "필요 조치",
                "구체적 방안",
                "기대 효과"
            ]
        else:
            return [
                "핵심 내용",
                "세부 설명",
                "결론"
            ]
    
    def get_korean_templates(self, domain: str) -> List[str]:
        templates = {
            "개인정보보호": [
                "개인정보보호법에 따라 {조치}를 수행해야 합니다.",
                "정보주체의 권리 보호를 위해 {방안}이 필요합니다.",
                "개인정보 처리 시 {원칙}을 준수해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 {조치}를 이행해야 합니다.",
                "전자금융거래의 안전성 확보를 위해 {방안}이 요구됩니다.",
                "이용자 보호를 위해 {조치}를 취해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계에 따라 {조치}를 구현해야 합니다.",
                "체계적인 보안 관리를 위해 {방안}이 필요합니다.",
                "보안 위험 관리를 위해 {조치}를 수행해야 합니다."
            ],
            "사이버보안": [
                "사이버 위협에 대응하기 위해 {조치}가 필요합니다.",
                "악성코드 탐지를 위해 {방안}을 적용해야 합니다.",
                "사이버 공격 방어를 위해 {조치}를 구축해야 합니다."
            ]
        }
        
        return templates.get(domain, [
            "관련 법령과 규정에 따라 {조치}를 수행해야 합니다.",
            "체계적인 관리를 위해 {방안}이 필요합니다."
        ])
    
    def validate_answer_quality(self, answer: str, domain: List[str]) -> Dict:
        quality_score = 0.0
        issues = []
        
        if len(answer) < 20:
            issues.append("답변이 너무 짧습니다")
        elif len(answer) > 1000:
            issues.append("답변이 너무 깁니다")
        else:
            quality_score += 0.3
        
        korean_ratio = len(re.findall(r'[가-힣]', answer)) / max(len(answer), 1)
        if korean_ratio > 0.5:
            quality_score += 0.3
        else:
            issues.append("한국어 비율이 낮습니다")
        
        if re.search(r'[\u4e00-\u9fff]', answer):
            issues.append("중국어 문자가 포함되어 있습니다")
        else:
            quality_score += 0.2
        
        professional_terms = ['법', '규정', '조치', '관리', '보안', '정책']
        if any(term in answer for term in professional_terms):
            quality_score += 0.2
        
        return {
            "quality_score": quality_score,
            "issues": issues,
            "is_acceptable": quality_score >= 0.6 and len(issues) == 0
        }