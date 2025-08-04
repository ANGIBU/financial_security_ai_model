# knowledge_base.py
"""
금융보안 특화 지식 베이스 및 문제 분류 시스템
FSKU 평가지표 기반 전문 지식 데이터베이스
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FinancialConcept:
    """금융보안 개념 정의"""
    name: str
    definition: str
    related_laws: List[str]
    key_points: List[str]
    common_mistakes: List[str]

class FinancialSecurityKnowledgeBase:
    """금융보안 전문 지식 베이스"""
    
    def __init__(self):
        self.concepts = self._build_concept_database()
        self.laws = self._build_law_database()
        self.question_patterns = self._build_question_patterns()
        self.common_answers = self._build_common_answers()
    
    def _build_concept_database(self) -> Dict[str, FinancialConcept]:
        """핵심 금융보안 개념 데이터베이스 구축"""
        concepts = {}
        
        # 개인정보보호
        concepts["개인정보보호"] = FinancialConcept(
            name="개인정보보호",
            definition="개인을 식별할 수 있는 정보의 처리에 관한 보호 체계",
            related_laws=["개인정보보호법", "신용정보법", "전자금융거래법"],
            key_points=[
                "개인정보 처리의 최소화 원칙",
                "정보주체의 동의 원칙", 
                "목적 외 이용·제공 금지",
                "개인정보 유출 시 즉시 신고 의무"
            ],
            common_mistakes=[
                "동의 없는 제3자 제공",
                "목적 외 이용",
                "보유기간 경과 후 미파기"
            ]
        )
        
        # 전자금융거래
        concepts["전자금융거래"] = FinancialConcept(
            name="전자금융거래",
            definition="전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래",
            related_laws=["전자금융거래법", "전자서명법"],
            key_points=[
                "전자적 장치 이용 필수",
                "금융기관과 이용자 간 계약",
                "보안절차 준수 의무",
                "거래내역 통지 의무"
            ],
            common_mistakes=[
                "전자적 장치 미사용 거래 포함",
                "보안절차 미준수"
            ]
        )
        
        # 암호화
        concepts["암호화"] = FinancialConcept(
            name="암호화",
            definition="정보를 제3자가 알아볼 수 없도록 변환하는 기술",
            related_laws=["정보통신망법", "개인정보보호법"],
            key_points=[
                "개인정보 전송 시 암호화 필수",
                "저장 시 암호화 권장",
                "암호화 알고리즘 선택 중요",
                "키 관리 체계 필수"
            ],
            common_mistakes=[
                "약한 암호화 알고리즘 사용",
                "키 관리 소홀"
            ]
        )
        
        # 접근제어
        concepts["접근제어"] = FinancialConcept(
            name="접근제어",
            definition="정보시스템 자원에 대한 접근을 통제하는 보안 기능",
            related_laws=["정보통신망법"],
            key_points=[
                "사용자 인증 필수",
                "권한 부여 원칙",
                "최소권한 원칙",
                "정기적 권한 검토"
            ],
            common_mistakes=[
                "과도한 권한 부여",
                "권한 검토 미실시"
            ]
        )
        
        # 사이버보안
        concepts["사이버보안"] = FinancialConcept(
            name="사이버보안",
            definition="사이버 공간에서의 위협으로부터 정보시스템을 보호하는 활동",
            related_laws=["정보보호산업법", "국가사이버보안법"],
            key_points=[
                "위협 탐지 및 대응",
                "보안 모니터링",
                "사고 대응 체계",
                "보안 교육 및 훈련"
            ],
            common_mistakes=[
                "실시간 모니터링 부족",
                "사고 대응 지연"
            ]
        )
        
        return concepts
    
    def _build_law_database(self) -> Dict[str, Dict]:
        """주요 법령 데이터베이스 구축"""
        laws = {}
        
        laws["개인정보보호법"] = {
            "목적": "개인정보의 처리 및 보호에 관한 사항을 정함",
            "주요조항": {
                "제15조": "개인정보의 수집·이용",
                "제17조": "개인정보의 제공",
                "제21조": "개인정보의 파기",
                "제22조의2": "만14세 미만 아동의 개인정보 처리",
                "제34조": "개인정보 유출 통지"
            },
            "벌칙": "5년 이하 징역 또는 5천만원 이하 벌금"
        }
        
        laws["전자금융거래법"] = {
            "목적": "전자금융거래의 법률관계를 명확히 하고 이용자를 보호",
            "주요조항": {
                "제2조": "정의",
                "제21조": "접근매체의 선정과 사용 및 관리",
                "제22조": "접근매체의 발급 및 관리",
                "제24조": "거래내역의 통지",
                "제44조": "과징금 등"
            },
            "벌칙": "3년 이하 징역 또는 3천만원 이하 벌금"
        }
        
        laws["정보통신망법"] = {
            "목적": "정보통신망의 이용촉진 및 정보보호에 관한 법률",
            "주요조항": {
                "제28조": "개인정보의 수집제한 등",
                "제29조": "개인정보의 처리제한",
                "제45조": "과징금",
                "제71조": "벌칙"
            },
            "벌칙": "5년 이하 징역 또는 5천만원 이하 벌금"
        }
        
        return laws
    
    def _build_question_patterns(self) -> Dict[str, List[str]]:
        """문제 유형별 패턴 분석"""
        patterns = {}
        
        patterns["정의_문제"] = [
            "정의로 가장 적절한 것은",
            "의미로 옳은 것은",
            "개념을 설명한 것은",
            "이란 무엇인가"
        ]
        
        patterns["법령_문제"] = [
            "법에 따르면",
            "규정에 의하면", 
            "조항에 해당하는",
            "벌칙은",
            "신고 기한은"
        ]
        
        patterns["기술_문제"] = [
            "암호화 알고리즘",
            "해킹 기법",
            "보안 솔루션",
            "기술적 특징"
        ]
        
        patterns["절차_문제"] = [
            "절차로 옳은 것은",
            "순서로 적절한",
            "단계는",
            "과정에서"
        ]
        
        patterns["부정형_문제"] = [
            "해당하지 않는",
            "적절하지 않은", 
            "옳지 않은",
            "틀린 것은",
            "제외한 것은"
        ]
        
        return patterns
    
    def _build_common_answers(self) -> Dict[str, str]:
        """일반적인 정답 패턴"""
        return {
            "개인정보_유출_신고": "즉시 또는 지체 없이",
            "암호화_필수": "개인정보 전송 시 필수",
            "동의_원칙": "사전 동의 필요",
            "최소수집_원칙": "꼭 필요한 최소한의 정보만",
            "목적외_이용": "원칙적으로 금지",
            "보유기간": "목적 달성 후 지체 없이 파기"
        }
    
    def analyze_question(self, question: str) -> Dict:
        """문제 분석 및 관련 지식 추출"""
        analysis = {
            "question_type": self._classify_question_type(question),
            "domain": self._identify_domain(question),
            "related_concepts": self._extract_related_concepts(question),
            "relevant_laws": self._extract_relevant_laws(question),
            "key_hints": self._extract_key_hints(question),
            "negative_question": self._is_negative_question(question)
        }
        return analysis
    
    def _classify_question_type(self, question: str) -> str:
        """문제 유형 분류"""
        question_lower = question.lower()
        
        for pattern_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    return pattern_type
        
        return "일반_문제"
    
    def _identify_domain(self, question: str) -> List[str]:
        """도메인 식별"""
        domains = []
        question_lower = question.lower()
        
        domain_keywords = {
            "개인정보보호": ["개인정보", "정보주체", "동의", "수집", "이용", "제공", "파기"],
            "전자금융": ["전자금융", "전자적장치", "접근매체", "거래", "인증", "비밀번호"],
            "사이버보안": ["해킹", "악성코드", "피싱", "파밍", "보안", "침입", "방화벽"],
            "암호화": ["암호화", "복호화", "키", "알고리즘", "해시", "전자서명"],
            "정보보호": ["정보보호", "보안정책", "접근제어", "감사", "모니터링"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["일반"]
    
    def _extract_related_concepts(self, question: str) -> List[str]:
        """관련 개념 추출"""
        related = []
        question_lower = question.lower()
        
        for concept_name in self.concepts.keys():
            if concept_name.lower() in question_lower:
                related.append(concept_name)
                
        return related
    
    def _extract_relevant_laws(self, question: str) -> List[str]:
        """관련 법령 추출"""
        relevant = []
        question_lower = question.lower()
        
        law_indicators = {
            "개인정보보호법": ["개인정보보호법", "개인정보", "정보주체"],
            "전자금융거래법": ["전자금융거래법", "전자금융", "접근매체"],
            "정보통신망법": ["정보통신망", "정보통신망법", "온라인"],
            "전자서명법": ["전자서명", "공인인증서", "디지털서명"]
        }
        
        for law, indicators in law_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                relevant.append(law)
                
        return relevant
    
    def _extract_key_hints(self, question: str) -> List[str]:
        """핵심 힌트 추출"""
        hints = []
        
        # 숫자 힌트
        numbers = re.findall(r'(\d+)(?:일|년|개월|시간|분)', question)
        if numbers:
            hints.extend([f"{num}단위_기간" for num in numbers])
        
        # 키워드 힌트
        key_phrases = [
            "즉시", "지체없이", "사전", "사후", "필수", "선택", 
            "의무", "권장", "금지", "허용", "최소", "최대"
        ]
        
        for phrase in key_phrases:
            if phrase in question:
                hints.append(phrase)
                
        return hints
    
    def _is_negative_question(self, question: str) -> bool:
        """부정형 문제 판별"""
        negative_indicators = [
            "해당하지 않는", "적절하지 않은", "옳지 않은", 
            "틀린", "잘못된", "부적절한", "제외한"
        ]
        
        return any(indicator in question for indicator in negative_indicators)
    
    def get_expert_knowledge(self, concept: str) -> Optional[FinancialConcept]:
        """전문 지식 조회"""
        return self.concepts.get(concept)
    
    def get_law_info(self, law_name: str) -> Optional[Dict]:
        """법령 정보 조회"""
        return self.laws.get(law_name)
    
    def generate_analysis_context(self, question: str) -> str:
        """문제 분석 컨텍스트 생성"""
        analysis = self.analyze_question(question)
        
        context = []
        
        # 문제 유형
        context.append(f"문제 유형: {analysis['question_type']}")
        
        # 관련 도메인
        if analysis['domain']:
            context.append(f"관련 분야: {', '.join(analysis['domain'])}")
        
        # 관련 개념
        if analysis['related_concepts']:
            for concept in analysis['related_concepts']:
                concept_info = self.get_expert_knowledge(concept)
                if concept_info:
                    context.append(f"\n[{concept}]")
                    context.append(f"정의: {concept_info.definition}")
                    context.append(f"핵심 포인트: {', '.join(concept_info.key_points[:2])}")
        
        # 관련 법령
        if analysis['relevant_laws']:
            context.append(f"\n관련 법령: {', '.join(analysis['relevant_laws'])}")
        
        # 부정형 문제 경고
        if analysis['negative_question']:
            context.append("\n⚠️ 부정형 문제: '해당하지 않는' 또는 '틀린' 것을 찾으세요!")
        
        return "\n".join(context)