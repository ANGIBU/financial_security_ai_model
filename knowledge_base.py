# knowledge_base.py
"""
금융보안 특화 지식 베이스 및 문제 분류 시스템 - 최적화 버전
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
    """금융보안 전문 지식 베이스 - 최적화 버전"""
    
    def __init__(self):
        self.concepts = self._build_concept_database()
        self.laws = self._build_law_database()
        self.question_patterns = self._build_question_patterns()
        self.domain_keywords = self._build_domain_keywords()
        
        # 성능 향상을 위한 컴파일된 패턴
        self.compiled_patterns = self._compile_patterns()
    
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
                "개인정보 유출 시 지체 없이 신고"
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
        
        # 정보보호관리체계
        concepts["ISMS"] = FinancialConcept(
            name="정보보호관리체계",
            definition="정보자산 보호를 위한 정책, 조직, 기술적 대책의 종합 체계",
            related_laws=["정보통신망법", "개인정보보호법"],
            key_points=[
                "관리체계 수립 및 운영",
                "위험평가 및 관리",
                "보호대책 구현",
                "지속적 개선"
            ],
            common_mistakes=[
                "일회성 구축으로 끝남",
                "형식적 운영"
            ]
        )
        
        # 금융보안
        concepts["금융보안"] = FinancialConcept(
            name="금융보안",
            definition="금융거래 및 금융정보의 안전성과 신뢰성 확보를 위한 보안 활동",
            related_laws=["전자금융거래법", "정보통신망법"],
            key_points=[
                "금융거래 안전성 확보",
                "금융정보 기밀성 유지",
                "이상거래 탐지",
                "사고 대응 체계"
            ],
            common_mistakes=[
                "사후 대응만 중시",
                "기술적 보안만 강조"
            ]
        )
        
        return concepts
    
    def _build_law_database(self) -> Dict[str, Dict]:
        """주요 법령 데이터베이스 구축 - 간소화"""
        laws = {}
        
        laws["개인정보보호법"] = {
            "목적": "개인정보의 처리 및 보호에 관한 사항을 정함",
            "주요조항": {
                "제15조": "개인정보의 수집·이용",
                "제17조": "개인정보의 제공",
                "제21조": "개인정보의 파기",
                "제34조": "개인정보 유출 통지 - 지체 없이"
            },
            "핵심원칙": ["최소수집", "목적명확", "동의필수", "안전관리"]
        }
        
        laws["전자금융거래법"] = {
            "목적": "전자금융거래의 법률관계를 명확히 하고 이용자를 보호",
            "주요조항": {
                "제2조": "전자금융거래 정의",
                "제21조": "접근매체의 선정과 사용 및 관리",
                "제24조": "거래내역의 통지"
            },
            "핵심원칙": ["전자적 장치", "안전성", "이용자 보호"]
        }
        
        return laws
    
    def _build_question_patterns(self) -> Dict[str, List[str]]:
        """문제 유형별 패턴 분석"""
        patterns = {
            "정의_문제": [
                r"정의로\s*(?:가장\s*)?적절한",
                r"의미로\s*옳은",
                r"개념.*설명",
                r"란\s*무엇"
            ],
            "법령_문제": [
                r"법.*따르면",
                r"규정.*의하면", 
                r"조항.*해당",
                r"법.*제\d+조"
            ],
            "부정형_문제": [
                r"해당하지\s*않는",
                r"적절하지\s*않은", 
                r"옳지\s*않은",
                r"틀린\s*것"
            ]
        }
        return patterns
    
    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """도메인별 핵심 키워드"""
        return {
            "개인정보보호": ["개인정보", "정보주체", "동의", "수집", "이용", "제공", "파기", "유출"],
            "전자금융": ["전자금융", "전자적장치", "접근매체", "거래", "인증", "전자서명"],
            "보안관리": ["보안", "관리", "통제", "정책", "절차", "체계", "ISMS"],
            "사이버보안": ["해킹", "악성코드", "침해", "취약점", "보안사고"],
            "암호화": ["암호화", "복호화", "키", "인증서", "전자서명"]
        }
    
    def _compile_patterns(self) -> Dict:
        """패턴 컴파일 (성능 향상)"""
        compiled = {}
        for pattern_type, patterns in self.question_patterns.items():
            compiled[pattern_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def analyze_question(self, question: str) -> Dict:
        """문제 분석 및 관련 지식 추출 - 최적화"""
        analysis = {
            "question_type": self._classify_question_type_fast(question),
            "domain": self._identify_domain_fast(question),
            "related_concepts": [],
            "relevant_laws": [],
            "key_hints": [],
            "negative_question": self._is_negative_question_fast(question)
        }
        
        # 필요시에만 상세 분석
        if analysis["question_type"] in ["법령_문제", "정의_문제"]:
            analysis["related_concepts"] = self._extract_related_concepts(question)
            analysis["relevant_laws"] = self._extract_relevant_laws_fast(question)
        
        return analysis
    
    def _classify_question_type_fast(self, question: str) -> str:
        """빠른 문제 유형 분류"""
        for pattern_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(question):
                    return pattern_type
        return "일반_문제"
    
    def _identify_domain_fast(self, question: str) -> List[str]:
        """빠른 도메인 식별"""
        domains = []
        question_lower = question.lower()
        
        # 핵심 키워드만 체크
        if "개인정보" in question_lower:
            domains.append("개인정보보호")
        if "전자금융" in question_lower or "전자적" in question_lower:
            domains.append("전자금융")
        if "보안" in question_lower and "관리" in question_lower:
            domains.append("보안관리")
        if "암호화" in question_lower:
            domains.append("암호화")
        
        return domains if domains else ["일반"]
    
    def _is_negative_question_fast(self, question: str) -> bool:
        """빠른 부정형 질문 판별"""
        negative_keywords = ['해당하지 않는', '적절하지 않은', '옳지 않은', '틀린', '잘못된']
        return any(keyword in question for keyword in negative_keywords)
    
    def _extract_related_concepts(self, question: str) -> List[str]:
        """관련 개념 추출"""
        related = []
        question_lower = question.lower()
        
        for concept_name in self.concepts.keys():
            if concept_name.lower() in question_lower:
                related.append(concept_name)
                
        return related
    
    def _extract_relevant_laws_fast(self, question: str) -> List[str]:
        """빠른 관련 법령 추출"""
        relevant = []
        question_lower = question.lower()
        
        if "개인정보" in question_lower:
            relevant.append("개인정보보호법")
        if "전자금융" in question_lower:
            relevant.append("전자금융거래법")
        if "정보통신망" in question_lower:
            relevant.append("정보통신망법")
            
        return relevant
    
    def get_expert_knowledge(self, concept: str) -> Optional[FinancialConcept]:
        """전문 지식 조회"""
        return self.concepts.get(concept)
    
    def get_law_info(self, law_name: str) -> Optional[Dict]:
        """법령 정보 조회"""
        return self.laws.get(law_name)
    
    def generate_analysis_context(self, question: str) -> str:
        """문제 분석 컨텍스트 생성 - 간소화"""
        analysis = self.analyze_question(question)
        
        if not analysis['relevant_laws'] and not analysis['related_concepts']:
            return ""  # 빈 컨텍스트로 빠른 처리
        
        context = []
        
        # 관련 법령 (중요한 것만)
        if analysis['relevant_laws']:
            law_info = []
            for law in analysis['relevant_laws'][:1]:  # 첫 번째 법령만
                info = self.get_law_info(law)
                if info:
                    law_info.append(f"{law}: {info['목적']}")
            if law_info:
                context.append("관련 법령: " + ", ".join(law_info))
        
        # 부정형 문제 경고
        if analysis['negative_question']:
            context.append("⚠️ 부정형 문제: '해당하지 않는' 것을 찾으세요!")
        
        return "\n".join(context)