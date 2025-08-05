# knowledge_base.py
"""
금융보안 지식 베이스
"""

import re
import hashlib
import json
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
    practical_applications: List[str]
    related_concepts: List[str]

@dataclass
class LegalArticle:
    """법령 조항"""
    law_name: str
    article_number: str
    title: str
    content: str
    key_keywords: List[str]
    practical_impact: str

class FinancialSecurityKnowledgeBase:
    """금융보안 지식 베이스"""
    
    def __init__(self):
        self.concepts = self._build_concept_database()
        self.laws = self._build_law_database()
        self.legal_articles = self._build_legal_articles_database()
        self.question_patterns = self._build_question_patterns()
        self.domain_keywords = self._build_domain_keywords()
        self.case_studies = self._build_case_studies()
        
        # 성능 최적화 캐시
        self.analysis_cache = {}
        self.pattern_cache = {}
        self.concept_cache = {}
        
        # 컴파일된 패턴들
        self.compiled_patterns = self._compile_all_patterns()
        
        # 지식 베이스 통계
        self.usage_stats = {
            "concept_queries": {},
            "domain_classifications": {},
            "pattern_matches": {},
            "cache_performance": {"hits": 0, "misses": 0}
        }
    
    def _build_concept_database(self) -> Dict[str, FinancialConcept]:
        """포괄적 금융보안 개념 데이터베이스"""
        concepts = {}
        
        # 개인정보보호
        concepts["개인정보보호"] = FinancialConcept(
            name="개인정보보호",
            definition="개인을 식별할 수 있는 정보의 수집, 이용, 제공, 보관, 파기 등 처리 전 과정에서의 보호",
            related_laws=["개인정보보호법", "신용정보법", "전자금융거래법", "정보통신망법"],
            key_points=[
                "개인정보 처리의 최소화 원칙",
                "정보주체의 동의 원칙",
                "목적 외 이용·제공 금지",
                "개인정보 유출 시 지체 없이 신고",
                "안전성 확보조치 의무",
                "개인정보 영향평가 실시"
            ],
            common_mistakes=[
                "동의 없는 제3자 제공",
                "목적 외 이용",
                "보유기간 경과 후 미파기",
                "안전성 확보조치 미흡",
                "유출 신고 지연"
            ],
            practical_applications=[
                "고객정보 수집 시 동의서 작성",
                "개인정보 처리방침 수립",
                "개인정보 보호책임자 지정",
                "정기적 개인정보 보호 교육"
            ],
            related_concepts=["정보주체 권리", "개인정보처리자", "안전성확보조치"]
        )
        
        # 전자금융거래
        concepts["전자금융거래"] = FinancialConcept(
            name="전자금융거래",
            definition="전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래",
            related_laws=["전자금융거래법", "전자서명법", "전자상거래법"],
            key_points=[
                "전자적 장치 이용 필수",
                "금융기관과 이용자 간 계약 관계",
                "보안절차 준수 의무",
                "거래내역 통지 의무",
                "손실부담 원칙",
                "접근매체 안전관리"
            ],
            common_mistakes=[
                "전자적 장치 미사용 거래 포함",
                "보안절차 미준수",
                "거래내역 통지 누락",
                "접근매체 관리 소홀"
            ],
            practical_applications=[
                "인터넷뱅킹 서비스",
                "모바일뱅킹 앱",
                "ATM 거래",
                "카드 결제 시스템"
            ],
            related_concepts=["접근매체", "전자서명", "공인인증서"]
        )
        
        # 정보보호관리체계
        concepts["ISMS"] = FinancialConcept(
            name="정보보호관리체계",
            definition="조직의 정보자산을 보호하기 위한 정책, 조직, 기술적 대책의 종합적 관리체계",
            related_laws=["정보통신망법", "개인정보보호법", "전자금융거래법"],
            key_points=[
                "관리체계 수립 및 운영",
                "위험분석 및 관리",
                "보호대책 구현",
                "사후관리",
                "지속적 개선",
                "최고경영자 책임"
            ],
            common_mistakes=[
                "일회성 구축으로 끝남",
                "형식적 운영",
                "위험분석 미흡",
                "지속적 개선 부족"
            ],
            practical_applications=[
                "보안정책 수립",
                "보안조직 구성",
                "보안시스템 구축",
                "보안교육 실시"
            ],
            related_concepts=["위험관리", "접근통제", "암호화", "침해사고대응"]
        )
        
        # 금융보안
        concepts["금융보안"] = FinancialConcept(
            name="금융보안",
            definition="금융거래 및 금융정보의 안전성과 신뢰성 확보를 위한 종합적 보안 활동",
            related_laws=["전자금융거래법", "정보통신망법", "개인정보보호법"],
            key_points=[
                "금융거래 안전성 확보",
                "금융정보 기밀성 유지",
                "이상거래 탐지 및 차단",
                "사고 대응 체계 구축",
                "고객 인증 강화",
                "보안기술 적용"
            ],
            common_mistakes=[
                "사후 대응 위주 운영",
                "기술적 보안만 강조",
                "고객 편의성 과도 추구",
                "보안교육 소홀"
            ],
            practical_applications=[
                "다중 인증 시스템",
                "실시간 이상거래 탐지",
                "보안카드 발급",
                "금융보안원 협력"
            ],
            related_concepts=["피싱", "파밍", "스미싱", "보이스피싱"]
        )
        
        # 암호화
        concepts["암호화"] = FinancialConcept(
            name="암호화",
            definition="정보의 기밀성과 무결성을 보장하기 위해 평문을 암호문으로 변환하는 기술",
            related_laws=["전자서명법", "정보통신망법", "개인정보보호법"],
            key_points=[
                "대칭키 암호화",
                "공개키 암호화",
                "해시 함수",
                "전자서명",
                "키 관리",
                "암호화 강도"
            ],
            common_mistakes=[
                "약한 암호화 알고리즘 사용",
                "키 관리 부실",
                "암호화 범위 제한",
                "성능 고려 미흡"
            ],
            practical_applications=[
                "데이터베이스 암호화",
                "통신 구간 암호화",
                "파일 암호화",
                "개인정보 암호화"
            ],
            related_concepts=["PKI", "인증서", "디지털서명", "해시"]
        )
        
        return concepts
    
    def _build_law_database(self) -> Dict[str, Dict]:
        """상세 법령 데이터베이스"""
        laws = {}
        
        laws["개인정보보호법"] = {
            "목적": "개인정보의 처리 및 보호에 관한 사항을 정하여 개인의 자유와 권리를 보호",
            "주요조항": {
                "제15조": "개인정보의 수집·이용",
                "제17조": "개인정보의 제공",
                "제21조": "개인정보의 파기",
                "제29조": "안전성 확보조치",
                "제34조": "개인정보 유출 통지 등",
                "제35조": "손해배상책임"
            },
            "핵심원칙": ["최소수집", "목적명확", "동의필수", "안전관리", "책임경영"],
            "처벌규정": {
                "5년 이하 징역": "동의 없는 민감정보 처리",
                "3년 이하 징역": "목적 외 이용·제공",
                "과태료": "안전성확보조치 위반"
            },
            "최근개정": "2020.8.5 시행",
            "관련기관": ["개인정보보호위원회", "개인정보보호 전문기관"]
        }
        
        laws["전자금융거래법"] = {
            "목적": "전자금융거래의 법률관계를 명확히 하고 전자금융업의 건전한 발전과 이용자 보호",
            "주요조항": {
                "제2조": "전자금융거래 정의",
                "제21조": "접근매체의 선정과 사용 및 관리",
                "제22조": "접근매체의 발급 등",
                "제23조": "접근매체의 이용 및 관리",
                "제24조": "거래내역의 통지",
                "제9조": "전자금융업무의 위탁"
            },
            "핵심원칙": ["전자적 장치", "안전성", "이용자 보호", "책임 분담"],
            "처벌규정": {
                "5년 이하 징역": "부정한 방법으로 전자금융거래",
                "3년 이하 징역": "접근매체 양도·대여",
                "과태료": "거래내역 통지 위반"
            },
            "최근개정": "2021.3.25 시행",
            "관련기관": ["금융위원회", "금융감독원", "금융보안원"]
        }
        
        laws["정보통신망법"] = {
            "목적": "정보통신망의 이용촉진 및 정보보호 등에 관한 법률",
            "주요조항": {
                "제47조": "정보보호 관리체계 인증",
                "제48조": "정보보호 안전진단",
                "제48조의3": "개인정보의 처리 제한",
                "제49조": "개인정보의 수집 제한 등"
            },
            "핵심원칙": ["정보보호", "개인정보보호", "이용자 보호"],
            "관련기관": ["방송통신위원회", "한국인터넷진흥원"]
        }
        
        return laws
    
    def _build_legal_articles_database(self) -> List[LegalArticle]:
        """법령 조항 데이터베이스"""
        articles = []
        
        # 개인정보보호법 주요 조항
        articles.extend([
            LegalArticle(
                law_name="개인정보보호법",
                article_number="제34조",
                title="개인정보 유출 통지 등",
                content="개인정보처리자는 개인정보가 유출된 사실을 안 때에는 지체 없이 해당 개인정보의 정보주체에게 다음 각 호의 사항을 알려야 한다",
                key_keywords=["유출", "통지", "지체없이", "정보주체"],
                practical_impact="개인정보 유출 시 즉시 통지 의무"
            ),
            LegalArticle(
                law_name="개인정보보호법",
                article_number="제29조",
                title="안전성 확보조치",
                content="개인정보처리자는 개인정보가 분실·도난·유출·위조·변조 또는 훼손되지 아니하도록 안전성 확보에 필요한 기술적·관리적 및 물리적 조치를 하여야 한다",
                key_keywords=["안전성확보조치", "기술적", "관리적", "물리적"],
                practical_impact="개인정보 보호를 위한 포괄적 조치 의무"
            )
        ])
        
        # 전자금융거래법 주요 조항
        articles.extend([
            LegalArticle(
                law_name="전자금융거래법",
                article_number="제2조",
                title="정의",
                content="전자금융거래란 전자적 장치를 통하여 금융상품과 서비스를 제공하고 이용하는 거래를 말한다",
                key_keywords=["전자금융거래", "전자적장치", "금융상품", "서비스"],
                practical_impact="전자금융거래의 법적 정의 제시"
            ),
            LegalArticle(
                law_name="전자금융거래법",
                article_number="제21조",
                title="접근매체의 선정과 사용 및 관리",
                content="금융회사등은 전자금융거래에 이용되는 접근매체를 안전하고 신뢰할 수 있는 것으로 선정하여야 한다",
                key_keywords=["접근매체", "안전", "신뢰", "선정"],
                practical_impact="접근매체의 안전성 확보 의무"
            )
        ])
        
        return articles
    
    def _build_question_patterns(self) -> Dict[str, List[str]]:
        """문제 패턴 분석"""
        patterns = {
            "정의_문제": [
                r"정의로\s*(?:가장\s*)?적절한",
                r"의미로\s*(?:가장\s*)?옳은",
                r"개념.*?(?:가장\s*)?적절한",
                r"란\s*무엇",
                r"라고?\s*할\s*수\s*있는\s*것",
                r"해당하는\s*것"
            ],
            "법령_문제": [
                r"법.*?(?:따르면|의하면)",
                r"규정.*?(?:따르면|의하면)",
                r"조항.*?해당",
                r"법.*?제\d+조",
                r"시행령.*?규정",
                r"기준.*?(?:따르면|의하면)"
            ],
            "부정형_문제": [
                r"해당하지\s*않는",
                r"적절하지\s*않은",
                r"옳지\s*않은",
                r"틀린\s*것",
                r"잘못된\s*것",
                r"관계없는\s*것",
                r"제외.*?것",
                r"아닌\s*것"
            ],
            "절차_문제": [
                r"절차.*?(?:적절한|옳은)",
                r"순서.*?(?:적절한|옳은)",
                r"단계.*?(?:적절한|옳은)",
                r"과정.*?(?:적절한|옳은)",
                r"방법.*?(?:적절한|옳은)"
            ],
            "사례_문제": [
                r"다음\s*상황",
                r"사례",
                r"경우",
                r"상황에서",
                r"예시",
                r"사건"
            ],
            "비교_문제": [
                r"차이점",
                r"공통점",
                r"비교",
                r"구분",
                r"분류",
                r"대비"
            ]
        }
        return patterns
    
    def _build_domain_keywords(self) -> Dict[str, List[str]]:
        """포괄적 도메인 키워드"""
        return {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보처리자", "동의", "수집", "이용", "제공", 
                "파기", "유출", "안전성확보조치", "개인정보보호책임자", "개인정보처리방침",
                "민감정보", "고유식별정보", "영상정보처리기기", "개인정보영향평가",
                "개인정보보호위원회", "개인정보보호법"
            ],
            "전자금융": [
                "전자금융거래", "전자적장치", "접근매체", "금융회사", "이용자", "거래내역",
                "전자서명", "공인인증서", "생체인증", "보안카드", "OTP", "SMS인증",
                "전자금융업", "전자지급수단", "선불전자지급수단", "전자화폐",
                "전자금융거래법", "금융위원회", "금융감독원"
            ],
            "정보보안": [
                "정보보호", "정보보안", "보안관리", "접근통제", "인증", "권한관리",
                "취약점", "위험관리", "보안정책", "보안절차", "보안교육", "보안감사",
                "침해사고", "사고대응", "복구", "보안시스템", "방화벽", "침입탐지",
                "ISMS", "정보보호관리체계", "ISO27001", "보안컨설팅"
            ],
            "암호화": [
                "암호화", "복호화", "암호알고리즘", "대칭키", "공개키", "개인키",
                "해시함수", "디지털서명", "전자서명", "PKI", "인증서", "CA",
                "키관리", "키분배", "암호화강도", "AES", "RSA", "SHA",
                "SSL", "TLS", "VPN", "암호모듈"
            ],
            "사이버보안": [
                "해킹", "크래킹", "악성코드", "바이러스", "웜", "트로이목마",
                "랜섬웨어", "스파이웨어", "피싱", "파밍", "스미싱", "보이스피싱",
                "DDoS", "APT", "제로데이", "사회공학", "내부자위협",
                "사이버범죄", "사이버테러", "정보유출", "개인정보침해"
            ],
            "법령": [
                "법", "법률", "시행령", "시행규칙", "조", "항", "호", "규정", "기준",
                "지침", "가이드라인", "표준", "원칙", "의무", "권리", "책임",
                "제재", "처벌", "과태료", "과징금", "형벌", "민사책임",
                "행정처분", "시정명령", "개선명령", "업무정지"
            ]
        }
    
    def _build_case_studies(self) -> Dict[str, List[Dict]]:
        """사례 연구 데이터베이스"""
        return {
            "개인정보_유출": [
                {
                    "title": "금융기관 고객정보 유출 사례",
                    "scenario": "해킹을 통한 개인정보 대량 유출",
                    "key_issues": ["유출 통지 지연", "안전성확보조치 미흡"],
                    "legal_consequences": ["과징금 부과", "집단소송"],
                    "lessons": ["즉시 통지의 중요성", "예방적 보안조치 필요성"]
                }
            ],
            "전자금융_사고": [
                {
                    "title": "전자금융거래 사고 사례",
                    "scenario": "접근매체 도용을 통한 부정거래",
                    "key_issues": ["손실부담", "거래내역 통지"],
                    "legal_consequences": ["분쟁조정", "손해배상"],
                    "lessons": ["접근매체 관리의 중요성", "이상거래 탐지 시스템 필요"]
                }
            ]
        }
    
    def _compile_all_patterns(self) -> Dict:
        """모든 패턴 컴파일"""
        compiled = {}
        for pattern_type, patterns in self.question_patterns.items():
            compiled[pattern_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def analyze_question(self, question: str) -> Dict:
        """문제 분석"""
        
        # 캐시 확인
        q_hash = hashlib.md5(question.encode()).hexdigest()[:16]
        if q_hash in self.analysis_cache:
            self.usage_stats["cache_performance"]["hits"] += 1
            return self.analysis_cache[q_hash]
        
        self.usage_stats["cache_performance"]["misses"] += 1
        
        analysis = {
            "question_type": self._classify_question_type(question),
            "domain": self._identify_domain(question),
            "complexity": self._assess_complexity(question),
            "related_concepts": self._extract_related_concepts(question),
            "relevant_laws": self._extract_relevant_laws(question),
            "key_hints": self._generate_key_hints(question),
            "negative_question": self._is_negative_question(question),
            "legal_articles": self._find_relevant_articles(question),
            "difficulty_indicators": self._identify_difficulty_indicators(question)
        }
        
        # 캐시 저장
        self.analysis_cache[q_hash] = analysis
        
        # 통계 업데이트
        self._update_usage_stats(analysis)
        
        return analysis
    
    def _classify_question_type(self, question: str) -> str:
        """문제 유형 분류"""
        for pattern_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(question):
                    self.usage_stats["pattern_matches"][pattern_type] = \
                        self.usage_stats["pattern_matches"].get(pattern_type, 0) + 1
                    return pattern_type
        return "일반_문제"
    
    def _identify_domain(self, question: str) -> List[str]:
        """도메인 식별"""
        domains = []
        question_lower = question.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > 0:
                # 키워드 밀도 계산
                density = score / len(keywords)
                domain_scores[domain] = {
                    "score": score,
                    "density": density,
                    "keywords": matched_keywords
                }
        
        # 점수 기반 정렬
        sorted_domains = sorted(domain_scores.items(), 
                              key=lambda x: (x[1]["score"], x[1]["density"]), 
                              reverse=True)
        
        # 상위 도메인들 선택
        for domain, info in sorted_domains:
            if info["score"] >= 2 or info["density"] >= 0.1:
                domains.append(domain)
        
        return domains if domains else ["일반"]
    
    def _assess_complexity(self, question: str) -> float:
        """복잡도 평가"""
        complexity_score = 0.0
        
        # 길이 기반 복잡도
        length = len(question)
        complexity_score += min(length / 2000, 0.25)
        
        # 구조적 복잡도
        line_count = question.count('\n')
        choice_count = len(re.findall(r'^\s*[1-5]\s*[.)]', question, re.MULTILINE))
        complexity_score += min((line_count + choice_count) / 15, 0.2)
        
        # 법령 관련 복잡도
        law_references = len(re.findall(r'법|조|항|규정|시행령', question))
        complexity_score += min(law_references / 8, 0.2)
        
        # 전문 용어 복잡도
        all_keywords = []
        for keywords in self.domain_keywords.values():
            all_keywords.extend(keywords)
        
        term_count = sum(1 for term in all_keywords if term in question)
        complexity_score += min(term_count / 10, 0.15)
        
        # 숫자 및 특수 문자 복잡도
        numbers = len(re.findall(r'\d+', question))
        special_chars = len(re.findall(r'[%@#$&*()]', question))
        complexity_score += min((numbers + special_chars) / 20, 0.1)
        
        # 부정형 보너스
        if self._is_negative_question(question):
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def _extract_related_concepts(self, question: str) -> List[str]:
        """관련 개념 추출"""
        related = []
        question_lower = question.lower()
        
        # 직접 매칭
        for concept_name in self.concepts.keys():
            if concept_name.lower() in question_lower:
                related.append(concept_name)
        
        # 키워드 기반 추론
        concept_keywords = {
            "개인정보보호": ["동의", "수집", "이용", "제공", "파기"],
            "전자금융거래": ["전자적", "거래", "접근매체", "인증"],
            "ISMS": ["관리체계", "위험", "보안정책", "접근통제"],
            "암호화": ["암호", "키", "해시", "전자서명"]
        }
        
        for concept, keywords in concept_keywords.items():
            if concept not in related:
                match_count = sum(1 for kw in keywords if kw in question_lower)
                if match_count >= 2:
                    related.append(concept)
        
        return related
    
    def _extract_relevant_laws(self, question: str) -> List[str]:
        """관련 법령 추출"""
        relevant = []
        question_lower = question.lower()
        
        # 직접 법령명 매칭
        for law_name in self.laws.keys():
            if law_name.replace("법", "") in question_lower:
                relevant.append(law_name)
        
        # 키워드 기반 법령 추론
        law_indicators = {
            "개인정보보호법": ["개인정보", "정보주체", "개인정보처리자", "동의"],
            "전자금융거래법": ["전자금융", "전자적장치", "접근매체", "전자금융업"],
            "정보통신망법": ["정보통신망", "정보보호", "개인정보처리"],
            "전자서명법": ["전자서명", "공인인증서", "인증기관"]
        }
        
        for law, indicators in law_indicators.items():
            if law not in relevant:
                match_count = sum(1 for indicator in indicators if indicator in question_lower)
                if match_count >= 1:
                    relevant.append(law)
        
        return relevant
    
    def _generate_key_hints(self, question: str) -> List[str]:
        """핵심 힌트 생성"""
        hints = []
        
        # 부정형 문제 힌트
        if self._is_negative_question(question):
            hints.append("부정형 문제: 틀린 것 또는 해당하지 않는 것을 찾으세요")
        
        # 정의 문제 힌트
        if "정의" in question or "의미" in question:
            hints.append("정의 문제: 법령상 정확한 정의를 확인하세요")
        
        # 법령 문제 힌트
        if re.search(r'법.*따르면', question):
            hints.append("법령 문제: 해당 법령의 조항을 정확히 적용하세요")
        
        # 절차 문제 힌트
        if any(word in question for word in ["절차", "순서", "단계"]):
            hints.append("절차 문제: 올바른 순서와 단계를 확인하세요")
        
        return hints
    
    def _find_relevant_articles(self, question: str) -> List[LegalArticle]:
        """관련 법령 조항 찾기"""
        relevant_articles = []
        question_lower = question.lower()
        
        for article in self.legal_articles:
            # 키워드 매칭
            keyword_matches = sum(1 for keyword in article.key_keywords 
                                if keyword.lower() in question_lower)
            
            if keyword_matches >= 1:
                relevant_articles.append(article)
        
        # 관련성 순으로 정렬
        relevant_articles.sort(key=lambda x: sum(1 for kw in x.key_keywords 
                                               if kw.lower() in question_lower), 
                             reverse=True)
        
        return relevant_articles[:3]  # 상위 3개만 반환
    
    def _identify_difficulty_indicators(self, question: str) -> List[str]:
        """난이도 지표 식별"""
        indicators = []
        
        # 난이도 지표
        if len(question) > 500:
            indicators.append("긴 문제")
        
        if question.count('\n') > 8:
            indicators.append("복잡한 구조")
        
        if len(re.findall(r'법|조|항', question)) > 3:
            indicators.append("다수 법령 참조")
        
        if re.search(r'사례|상황|예시', question):
            indicators.append("사례 기반")
        
        if re.search(r'비교|차이|구분', question):
            indicators.append("비교 분석")
        
        return indicators
    
    def _is_negative_question(self, question: str) -> bool:
        """부정형 질문 판별"""
        negative_indicators = [
            r"해당하지\s*않는", r"적절하지\s*않은", r"옳지\s*않은",
            r"틀린\s*것", r"잘못된\s*것", r"부적절한", 
            r"제외.*?것", r"아닌\s*것", r"관계없는\s*것",
            r"거짓인\s*것", r"맞지\s*않는", r"무관한\s*것"
        ]
        
        for indicator in negative_indicators:
            if re.search(indicator, question, re.IGNORECASE):
                return True
        
        return False
    
    def get_expert_knowledge(self, concept: str) -> Optional[FinancialConcept]:
        """전문 지식 조회"""
        cache_key = f"concept_{concept}"
        
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        result = self.concepts.get(concept)
        if result:
            self.concept_cache[cache_key] = result
            
            # 통계 업데이트
            if concept not in self.usage_stats["concept_queries"]:
                self.usage_stats["concept_queries"][concept] = 0
            self.usage_stats["concept_queries"][concept] += 1
        
        return result
    
    def get_law_info(self, law_name: str) -> Optional[Dict]:
        """법령 정보 조회"""
        return self.laws.get(law_name)
    
    def generate_analysis_context(self, question: str) -> str:
        """분석 컨텍스트 생성"""
        analysis = self.analyze_question(question)
        
        if not analysis['relevant_laws'] and not analysis['related_concepts']:
            return ""
        
        context_parts = []
        
        # 관련 법령 정보
        if analysis['relevant_laws']:
            law_info = []
            for law in analysis['relevant_laws'][:2]:  # 최대 2개
                info = self.get_law_info(law)
                if info:
                    law_info.append(f"{law}: {info['목적']}")
            if law_info:
                context_parts.append("관련 법령:\n" + "\n".join(law_info))
        
        # 핵심 힌트
        if analysis['key_hints']:
            context_parts.append("중요 사항:\n" + "\n".join(f"- {hint}" for hint in analysis['key_hints']))
        
        # 관련 조항
        if analysis['legal_articles']:
            article_info = []
            for article in analysis['legal_articles'][:1]:  # 최대 1개
                article_info.append(f"{article.law_name} {article.article_number}: {article.practical_impact}")
            if article_info:
                context_parts.append("관련 조항:\n" + "\n".join(article_info))
        
        return "\n\n".join(context_parts)
    
    def _update_usage_stats(self, analysis: Dict):
        """사용 통계 업데이트"""
        # 도메인 분류 통계
        for domain in analysis["domain"]:
            if domain not in self.usage_stats["domain_classifications"]:
                self.usage_stats["domain_classifications"][domain] = 0
            self.usage_stats["domain_classifications"][domain] += 1
    
    def get_knowledge_base_report(self) -> Dict:
        """지식 베이스 보고서"""
        total_queries = sum(self.usage_stats["concept_queries"].values())
        cache_total = self.usage_stats["cache_performance"]["hits"] + self.usage_stats["cache_performance"]["misses"]
        
        return {
            "total_concepts": len(self.concepts),
            "total_laws": len(self.laws),
            "total_articles": len(self.legal_articles),
            "cache_hit_rate": self.usage_stats["cache_performance"]["hits"] / max(cache_total, 1),
            "most_queried_concept": max(self.usage_stats["concept_queries"].items(), 
                                      key=lambda x: x[1])[0] if self.usage_stats["concept_queries"] else "없음",
            "domain_distribution": self.usage_stats["domain_classifications"],
            "pattern_usage": self.usage_stats["pattern_matches"]
        }
    
    def cleanup(self):
        """리소스 정리"""
        # 통계 출력
        if self.usage_stats["concept_queries"]:
            print(f"\n=== 지식 베이스 통계 ===")
            total_cache = self.usage_stats["cache_performance"]["hits"] + self.usage_stats["cache_performance"]["misses"]
            if total_cache > 0:
                hit_rate = self.usage_stats["cache_performance"]["hits"] / total_cache
                print(f"캐시 히트율: {hit_rate:.2%}")
            
            if self.usage_stats["concept_queries"]:
                most_used = max(self.usage_stats["concept_queries"].items(), key=lambda x: x[1])
                print(f"주요 개념: {most_used[0]} ({most_used[1]}회)")
        
        # 캐시 정리
        self.analysis_cache.clear()
        self.pattern_cache.clear()
        self.concept_cache.clear()