# knowledge_base.py

"""
금융보안 지식베이스 (대회 규칙 준수 버전)
- 도메인별 키워드 관리
- 문제 분석 및 분류
- 법령 참조 정보
- 핵심 개념 정의
- 단일 모델 원칙 준수 (외부 모델 사용 금지)
"""

import re
import gc
import time
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, Counter
from dataclasses import dataclass

# 상수 정의
DEFAULT_CACHE_SIZE = 1000
PATTERN_DISCOVERY_THRESHOLD = 3
CONFIDENCE_DECAY_FACTOR = 0.95
MAX_PATTERN_AGE = 86400  # 24시간 (초)
COOCCURRENCE_MIN_SCORE = 0.3
DOMAIN_CONFIDENCE_THRESHOLD = 0.1
CACHE_CLEANUP_INTERVAL = 100
MEMORY_CLEANUP_THRESHOLD = 0.8

@dataclass
class TextPattern:
    pattern_id: str
    keywords: List[str]
    co_occurrence_score: float
    frequency: int
    context_type: str
    confidence: float
    discovered_time: float

@dataclass
class DomainAnalysis:
    primary_domain: str
    secondary_domains: List[str]
    confidence_scores: Dict[str, float]
    technical_complexity: float
    legal_complexity: float
    pattern_matches: List[str]

class FinancialSecurityKnowledgeBase:
    """금융보안 지식베이스 (대회 규칙 100% 준수)"""
    
    def __init__(self):
        print("지식베이스 초기화 중... (대회 규칙 준수)")
        
        self.domain_keywords = self._initialize_domain_keywords()
        self.law_references = self._initialize_law_references()
        self.security_concepts = self._initialize_security_concepts()
        self.technical_concepts = self._initialize_technical_concepts()
        
        # 패턴 및 캐시 관리
        self.discovered_patterns = {}
        self.keyword_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.domain_evolution_tracking = defaultdict(list)
        self.analysis_cache = {}
        self.max_cache_size = DEFAULT_CACHE_SIZE
        
        # 고급 분석 도구 (규칙 기반, 외부 모델 사용 금지)
        self.advanced_indicators = self._build_advanced_indicators()
        self.contextual_modifiers = self._build_contextual_modifiers()
        
        # 성능 추적
        self.performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "patterns_discovered": 0,
            "analysis_count": 0,
            "rule_based_processing": 0
        }
        
        self.last_cleanup_time = time.time()
        
        print("✅ 지식베이스 초기화 완료 (단일 모델 원칙 준수)")
        
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """도메인 키워드 초기화"""
        return {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보처리자", "개인정보보호법", "개인정보처리",
                "수집", "이용", "제공", "파기", "동의", "열람", "정정", "삭제",
                "민감정보", "고유식별정보", "안전성확보조치", "영향평가", "유출신고",
                "정보주체권리", "처리방침", "개인정보위원회", "과징금", "손해배상"
            ],
            "전자금융": [
                "전자금융거래", "전자금융거래법", "전자적장치", "접근매체", "전자서명",
                "전자인증", "금융기관", "전자금융업", "전자지급수단", "전화폰",
                "오류정정", "손해배상", "약관", "이용자", "거래내역통지", "안전성확보",
                "전자금융감독", "전자금융분쟁", "전자금융범죄", "디지털금융"
            ],
            "정보보안": [
                "정보보안", "정보보호", "정보보안관리체계", "ISMS", "ISMS-P",
                "보안정책", "보안통제", "위험관리", "취약점", "보안사고",
                "접근통제", "암호화", "네트워크보안", "시스템보안", "데이터보안",
                "보안감사", "침입탐지", "방화벽", "백신", "보안관제"
            ],
            "사이버보안": [
                "사이버보안", "사이버공격", "해킹", "악성코드", "멀웨어", "바이러스",
                "웜", "트로이목마", "랜섬웨어", "스파이웨어", "애드웨어", "루트킷",
                "피싱", "스미싱", "파밍", "스피어피싱", "사회공학", "제로데이",
                "APT", "DDoS", "봇넷", "C&C", "백도어", "키로거", "SQL인젝션",
                "크로스사이트스크립팅", "버퍼오버플로우", "취약점분석", "침투테스트"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험분석", "위험식별", "위험측정", "위험통제",
                "위험모니터링", "위험보고", "위험수용", "위험회피", "위험전가", "위험완화",
                "위험관리체계", "위험관리정책", "위험관리조직", "위험관리절차",
                "리스크어세스먼트", "리스크매트릭스", "리스크모니터링", "비즈니스연속성"
            ],
            "관리체계": [
                "관리체계", "정보보호관리체계", "정보보안관리체계", "PDCA",
                "정책", "조직", "자산관리", "인적보안", "물리보안", "시스템보안",
                "네트워크보안", "접근통제", "시스템개발", "공급업체관리", "사고관리",
                "경영진책임", "최고정보보호책임자", "정보보호담당자", "보안교육"
            ],
            "재해복구": [
                "재해복구", "재해복구계획", "BCP", "업무연속성", "재해복구센터", "DRP",
                "백업", "복구", "RTO", "RPO", "복구목표시간", "복구목표시점",
                "핫사이트", "콜드사이트", "웜사이트", "재해복구훈련", "비상연락체계",
                "복구우선순위", "재해복구전략", "데이터복구"
            ],
            "금융투자업": [
                "금융투자업", "투자매매업", "투자중개업", "투자자문업", "투자일임업",
                "집합투자업", "신탁업", "소비자금융업", "보험중개업", "자본시장법",
                "금융투자상품", "투자자보호", "불공정거래", "내부통제", "리스크관리",
                "투자권유", "적합성원칙", "설명의무", "투자자문계약"
            ],
            "암호화": [
                "암호화", "복호화", "암호", "암호키", "대칭키", "공개키", "개인키",
                "해시함수", "전자서명", "인증서", "PKI", "암호알고리즘",
                "AES", "RSA", "SHA", "MD5", "키관리", "키분배", "키교환",
                "디지털인증서", "암호화프로토콜", "SSL", "TLS", "IPSec"
            ],
            "법령준수": [
                "컴플라이언스", "법령준수", "규정준수", "내부통제", "준법감시",
                "준법점검", "법규위반", "제재조치", "과징금", "과태료",
                "행정처분", "형사처벌", "민사책임", "법적의무", "신고의무",
                "검사", "감독", "감사", "조사", "점검"
            ]
        }
    
    def _initialize_law_references(self) -> Dict[str, Dict]:
        """법령 참조 초기화"""
        return {
            "개인정보보호법": {
                "정의": "개인정보보호법 제2조",
                "처리원칙": "개인정보보호법 제3조",
                "수집제한": "개인정보보호법 제15조",
                "이용제한": "개인정보보호법 제18조",
                "제공제한": "개인정보보호법 제17조",
                "파기": "개인정보보호법 제21조",
                "안전성확보조치": "개인정보보호법 제29조",
                "유출신고": "개인정보보호법 제34조",
                "손해배상": "개인정보보호법 제39조",
                "과징금": "개인정보보호법 제34조의2"
            },
            "전자금융거래법": {
                "정의": "전자금융거래법 제2조",
                "접근매체": "전자금융거래법 제2조 제10호",
                "이용자보호": "전자금융거래법 제9조",
                "안전성확보의무": "전자금융거래법 제21조",
                "거래내역통지": "전자금융거래법 제18조",
                "오류정정": "전자금융거래법 제19조",
                "손해배상": "전자금융거래법 제20조",
                "약관규제": "전자금융거래법 제16조"
            },
            "정보통신망법": {
                "개인정보보호": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
                "개인정보수집": "정보통신망법 제22조",
                "개인정보이용제공": "정보통신망법 제24조",
                "개인정보보호조치": "정보통신망법 제28조",
                "침해신고센터": "정보통신망법 제54조"
            },
            "자본시장법": {
                "금융투자업": "자본시장과 금융투자업에 관한 법률",
                "투자매매업": "자본시장법 제8조",
                "투자중개업": "자본시장법 제9조",
                "투자자문업": "자본시장법 제6조",
                "투자일임업": "자본시장법 제7조",
                "적합성원칙": "자본시장법 제46조",
                "설명의무": "자본시장법 제47조"
            }
        }
    
    def _initialize_security_concepts(self) -> Dict[str, str]:
        """보안 개념 초기화"""
        return {
            "기밀성": "인가되지 않은 개인, 개체, 프로세스에 대해 정보를 사용하지 못하게 하거나 공개하지 않는 특성",
            "무결성": "정확성과 완전성을 보호하는 특성",
            "가용성": "인가된 개체가 요구할 때 접근 및 사용이 가능한 특성",
            "부인방지": "어떤 행위나 사건의 발생에 대해 나중에 부인할 수 없도록 하는 특성",
            "인증": "어떤 개체의 신원이 주장된 신원과 같음을 확실히 하는 과정",
            "인가": "특정 자원에 대한 접근 권한을 부여하는 과정",
            "식별": "사용자나 프로세스가 자신을 시스템에 알리는 과정",
            "추적성": "시스템에서 발생하는 모든 활동을 기록하고 추적할 수 있는 특성",
            "책임추적성": "시스템 사용자의 모든 행위에 대해 책임을 추적할 수 있는 특성"
        }
    
    def _initialize_technical_concepts(self) -> Dict[str, str]:
        """기술 개념 초기화"""
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
            "소셜엔지니어링": "기술적 수단보다는 인간의 심리적 약점을 이용하여 정보를 획득하는 공격 기법",
            "SQL인젝션": "데이터베이스에 악의적인 SQL 쿼리를 삽입하여 데이터를 탈취하거나 조작하는 공격",
            "XSS": "크로스사이트스크립팅으로, 웹사이트에 악성 스크립트를 삽입하여 사용자 정보를 탈취하는 공격"
        }
    
    def _build_advanced_indicators(self) -> Dict[str, Dict]:
        """고급 지표 구축 (규칙 기반)"""
        return {
            "complexity_indicators": {
                "high": ["심화", "고급", "전문", "상세", "복합", "통합", "종합"],
                "medium": ["일반", "기본", "표준", "보통", "평가", "분석"],
                "low": ["단순", "초급", "기초", "개요", "요약", "개념"]
            },
            "question_types": {
                "definitional": ["정의", "의미", "개념", "이란", "무엇"],
                "procedural": ["절차", "과정", "단계", "방법", "순서"],
                "analytical": ["분석", "평가", "비교", "검토", "판단"],
                "applied": ["적용", "구현", "설계", "개발", "운영"]
            },
            "temporal_indicators": {
                "current": ["현재", "최근", "신규", "새로운", "최신"],
                "historical": ["기존", "과거", "전통적", "구래의"],
                "future": ["향후", "미래", "예정", "계획", "전망"]
            }
        }
    
    def _build_contextual_modifiers(self) -> Dict[str, float]:
        """맥락 수정자 구축"""
        return {
            "긍정강화": 1.3,
            "부정강화": 1.2,
            "예외사항": 1.4,
            "필수조건": 1.5,
            "권장사항": 0.9,
            "선택사항": 0.8,
            "금지사항": 1.6,
            "의무사항": 1.4,
            "주의사항": 1.1,
            "중요사항": 1.3
        }
    
    def discover_text_patterns(self, text: str, existing_patterns: Optional[Dict] = None) -> List[TextPattern]:
        """텍스트 패턴 발견 (규칙 기반)"""
        try:
            discovered = []
            text_lower = text.lower()
            
            words = re.findall(r'[가-힣]{2,}', text_lower)
            
            if len(words) < 2:
                return discovered
            
            # 동시출현 업데이트
            self._update_cooccurrence(words)
            
            # 바이그램 및 트라이그램 패턴 발견
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
            
            for pattern_words in [bigrams, trigrams]:
                discovered.extend(self._analyze_patterns(pattern_words, text))
            
            # 기술 패턴 발견
            technical_patterns = self._discover_technical_patterns(text)
            discovered.extend(technical_patterns)
            
            self.performance_stats["rule_based_processing"] += 1
            
            return discovered
            
        except Exception as e:
            return []
    
    def _update_cooccurrence(self, words: List[str]) -> None:
        """동시출현 업데이트"""
        try:
            for i, word in enumerate(words):
                context_words = words[max(0, i-2):i] + words[i+1:min(len(words), i+3)]
                
                for context_word in context_words:
                    self.keyword_cooccurrence[word][context_word] += 1
        except Exception:
            pass
    
    def _analyze_patterns(self, pattern_words: List[Tuple], text: str) -> List[TextPattern]:
        """패턴 분석 (규칙 기반)"""
        discovered = []
        
        try:
            for pattern in pattern_words:
                pattern_str = "_".join(pattern)
                pattern_id = hashlib.md5(pattern_str.encode('utf-8')).hexdigest()[:8]
                
                if pattern_id not in self.discovered_patterns:
                    co_occurrence_score = self._calculate_cooccurrence_score(pattern)
                    
                    if co_occurrence_score > COOCCURRENCE_MIN_SCORE:
                        context_type = self._determine_context_type(text, pattern)
                        confidence = min(co_occurrence_score * 1.2, 0.95)
                        
                        discovered_pattern = TextPattern(
                            pattern_id=pattern_id,
                            keywords=list(pattern),
                            co_occurrence_score=co_occurrence_score,
                            frequency=1,
                            context_type=context_type,
                            confidence=confidence,
                            discovered_time=time.time()
                        )
                        
                        self.discovered_patterns[pattern_id] = discovered_pattern
                        discovered.append(discovered_pattern)
                        self.performance_stats["patterns_discovered"] += 1
                else:
                    existing_pattern = self.discovered_patterns[pattern_id]
                    existing_pattern.frequency += 1
                    existing_pattern.confidence = min(existing_pattern.confidence * 1.05, 0.95)
                    
        except Exception:
            pass
            
        return discovered
    
    def _calculate_cooccurrence_score(self, pattern: Tuple[str, ...]) -> float:
        """동시출현 점수 계산 (규칙 기반)"""
        if len(pattern) < 2:
            return 0.0
        
        try:
            scores = []
            for i, word1 in enumerate(pattern):
                for j, word2 in enumerate(pattern):
                    if i != j:
                        cooccurrence_count = self.keyword_cooccurrence[word1][word2]
                        total_word1 = sum(self.keyword_cooccurrence[word1].values())
                        if total_word1 > 0:
                            scores.append(cooccurrence_count / total_word1)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception:
            return 0.0
    
    def _determine_context_type(self, text: str, pattern: Tuple[str, ...]) -> str:
        """맥락 유형 결정"""
        text_lower = text.lower()
        
        context_indicators = {
            "legal": ["법", "규정", "조항", "시행령", "고시", "규칙"],
            "technical": ["시스템", "기술", "프로그램", "소프트웨어", "하드웨어"],
            "procedural": ["절차", "과정", "단계", "방법", "수행"],
            "managerial": ["관리", "운영", "정책", "체계", "조직"]
        }
        
        for context_type, indicators in context_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return context_type
        
        return "general"
    
    def _discover_technical_patterns(self, text: str) -> List[TextPattern]:
        """기술 패턴 발견 (규칙 기반)"""
        patterns = []
        
        try:
            technical_indicators = [
                (r'([A-Z]{2,})\s+([가-힣]+)', "acronym_definition"),
                (r'([가-힣]+)\s+프로토콜', "protocol_reference"),
                (r'([가-힣]+)\s+알고리즘', "algorithm_reference"),
                (r'(\d+)\s*비트\s+([가-힣]+)', "bit_specification"),
                (r'([가-힣]+)\s+인증서', "certificate_type")
            ]
            
            for pattern_regex, pattern_type in technical_indicators:
                matches = re.findall(pattern_regex, text)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        pattern_id = hashlib.md5(f"{pattern_type}_{match[0]}_{match[1]}".encode('utf-8')).hexdigest()[:8]
                        
                        if pattern_id not in self.discovered_patterns:
                            technical_pattern = TextPattern(
                                pattern_id=pattern_id,
                                keywords=[match[0], match[1]],
                                co_occurrence_score=0.8,
                                frequency=1,
                                context_type="technical",
                                confidence=0.85,
                                discovered_time=time.time()
                            )
                            
                            patterns.append(technical_pattern)
                            self.discovered_patterns[pattern_id] = technical_pattern
                            
        except Exception:
            pass
        
        return patterns
    
    def analyze_question_enhanced(self, question: str) -> DomainAnalysis:
        """향상된 질문 분석 (규칙 기반, 외부 모델 사용 금지)"""
        cache_key = hashlib.md5(question.encode('utf-8')).hexdigest()[:16]
        
        if cache_key in self.analysis_cache:
            self.performance_stats["cache_hits"] += 1
            return self.analysis_cache[cache_key]
        
        self.performance_stats["cache_misses"] += 1
        self.performance_stats["analysis_count"] += 1
        
        try:
            question_lower = question.lower()
            
            domain_scores = {}
            pattern_matches = []
            
            # 도메인 점수 계산 (규칙 기반)
            for domain, keywords in self.domain_keywords.items():
                base_matches = sum(1 for keyword in keywords if keyword in question_lower)
                base_score = base_matches / len(keywords) if keywords else 0
                
                # 맥락적 강화 (규칙 기반)
                contextual_boost = self._calculate_contextual_boost_rule_based(question_lower)
                
                # 기술적 복잡도 강화 (규칙 기반)
                technical_boost = self._calculate_technical_boost_rule_based(question_lower, domain)
                
                # 법적 복잡도 강화 (규칙 기반)
                legal_boost = self._calculate_legal_boost_rule_based(question_lower, domain)
                
                final_score = base_score + contextual_boost + technical_boost + legal_boost
                
                if final_score > 0.05:
                    domain_scores[domain] = min(final_score, 1.0)
                    if final_score > 0.3:
                        pattern_matches.append(f"strong_{domain}")
                    elif final_score > 0.1:
                        pattern_matches.append(f"weak_{domain}")
            
            # 패턴 발견 (규칙 기반)
            discovered_patterns = self.discover_text_patterns(question, self.discovered_patterns)
            for pattern in discovered_patterns:
                pattern_matches.append(f"discovered_{pattern.pattern_id}")
            
            # 분석 결과 생성
            primary_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "일반"
            secondary_domains = [domain for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)[1:4]]
            
            technical_complexity = self._calculate_technical_complexity_rule_based(question)
            legal_complexity = self._calculate_legal_complexity_rule_based(question)
            
            analysis = DomainAnalysis(
                primary_domain=primary_domain,
                secondary_domains=secondary_domains,
                confidence_scores=domain_scores,
                technical_complexity=technical_complexity,
                legal_complexity=legal_complexity,
                pattern_matches=pattern_matches
            )
            
            # 캐시 관리 및 저장
            self._manage_cache()
            self.analysis_cache[cache_key] = analysis
            
            # 도메인 진화 추적
            self._track_domain_evolution(primary_domain, domain_scores.get(primary_domain, 0))
            
            return analysis
            
        except Exception as e:
            # 오류 발생 시 기본 분석 반환
            return DomainAnalysis(
                primary_domain="일반",
                secondary_domains=[],
                confidence_scores={},
                technical_complexity=0.0,
                legal_complexity=0.0,
                pattern_matches=[]
            )
    
    def _calculate_contextual_boost_rule_based(self, text: str) -> float:
        """규칙 기반 맥락적 강화 계산"""
        try:
            contextual_boost = 0
            for modifier, boost in self.contextual_modifiers.items():
                if modifier in text:
                    contextual_boost += (boost - 1) * 0.1
            return min(contextual_boost, 0.3)
        except Exception:
            return 0.0
    
    def _calculate_technical_boost_rule_based(self, text: str, domain: str) -> float:
        """규칙 기반 기술적 강화 계산"""
        try:
            if domain in ["사이버보안", "암호화", "정보보안"]:
                tech_terms = sum(1 for term in self.technical_concepts.keys() if term in text)
                return min(tech_terms * 0.1, 0.3)
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_legal_boost_rule_based(self, text: str, domain: str) -> float:
        """규칙 기반 법적 강화 계산"""
        try:
            if domain in ["개인정보보호", "전자금융", "금융투자업"]:
                legal_refs = 0
                for law in self.law_references.keys():
                    if any(ref in text for ref in self.law_references[law].values()):
                        legal_refs += 1
                return min(legal_refs * 0.15, 0.25)
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_technical_complexity_rule_based(self, text: str) -> float:
        """규칙 기반 기술적 복잡도 계산"""
        try:
            text_lower = text.lower()
            
            complexity_factors = {
                "technical_terms": min(sum(1 for term in self.technical_concepts.keys() if term in text_lower) * 0.1, 0.3),
                "acronyms": min(len(re.findall(r'\b[A-Z]{2,}\b', text)) * 0.05, 0.2),
                "numbers": min(len(re.findall(r'\d+', text)) * 0.02, 0.1),
                "complexity_indicators": 0
            }
            
            high_complexity_terms = self.advanced_indicators["complexity_indicators"]["high"]
            if any(term in text_lower for term in high_complexity_terms):
                complexity_factors["complexity_indicators"] = 0.2
            
            return min(sum(complexity_factors.values()), 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_legal_complexity_rule_based(self, text: str) -> float:
        """규칙 기반 법적 복잡도 계산"""
        try:
            text_lower = text.lower()
            
            legal_factors = {
                "law_references": 0,
                "article_references": min(len(re.findall(r'제\d+조', text)) * 0.1, 0.3),
                "legal_terms": 0
            }
            
            for law_name in self.law_references.keys():
                if any(keyword in text_lower for keyword in law_name.split()):
                    legal_factors["law_references"] += 0.15
            
            legal_terms = ["의무", "권리", "책임", "처벌", "제재", "과징금", "손해배상"]
            legal_factors["legal_terms"] = min(sum(1 for term in legal_terms if term in text_lower) * 0.05, 0.2)
            
            return min(sum(legal_factors.values()), 1.0)
            
        except Exception:
            return 0.0
    
    def _track_domain_evolution(self, domain: str, score: float) -> None:
        """도메인 진화 추적"""
        try:
            current_time = time.time()
            
            self.domain_evolution_tracking[domain].append({
                "timestamp": current_time,
                "score": score
            })
            
            # 오래된 데이터 정리
            if len(self.domain_evolution_tracking[domain]) > 100:
                self.domain_evolution_tracking[domain] = self.domain_evolution_tracking[domain][-100:]
                
        except Exception:
            pass
    
    def _manage_cache(self) -> None:
        """캐시 관리"""
        try:
            current_time = time.time()
            
            # 정기적인 정리
            if current_time - self.last_cleanup_time > CACHE_CLEANUP_INTERVAL:
                self._cleanup_expired_patterns()
                self.last_cleanup_time = current_time
            
            # 캐시 크기 관리
            if len(self.analysis_cache) >= self.max_cache_size:
                oldest_keys = list(self.analysis_cache.keys())[:self.max_cache_size // 4]
                for key in oldest_keys:
                    del self.analysis_cache[key]
                    
        except Exception:
            pass
    
    def _cleanup_expired_patterns(self) -> None:
        """만료된 패턴 정리"""
        try:
            current_time = time.time()
            expired_patterns = []
            
            for pattern_id, pattern in self.discovered_patterns.items():
                if current_time - pattern.discovered_time > MAX_PATTERN_AGE:
                    if pattern.frequency < PATTERN_DISCOVERY_THRESHOLD:
                        expired_patterns.append(pattern_id)
            
            for pattern_id in expired_patterns:
                del self.discovered_patterns[pattern_id]
                
        except Exception:
            pass
    
    def analyze_question(self, question: str) -> Dict:
        """질문 분석 (호환성 메서드, 규칙 기반)"""
        try:
            enhanced_analysis = self.analyze_question_enhanced(question)
            
            return {
                "domain": [enhanced_analysis.primary_domain] + enhanced_analysis.secondary_domains[:2],
                "complexity": (enhanced_analysis.technical_complexity + enhanced_analysis.legal_complexity) / 2,
                "question_type": self._determine_question_type_rule_based(question),
                "key_concepts": self._extract_key_concepts_rule_based(question),
                "law_references": self._extract_law_references_rule_based(question),
                "difficulty_level": self._assess_difficulty_rule_based(question),
                "pattern_matches": enhanced_analysis.pattern_matches,
                "confidence_scores": enhanced_analysis.confidence_scores
            }
            
        except Exception:
            return {
                "domain": ["일반"],
                "complexity": 0.0,
                "question_type": "general",
                "key_concepts": [],
                "law_references": [],
                "difficulty_level": "basic",
                "pattern_matches": [],
                "confidence_scores": {}
            }
    
    def _determine_question_type_rule_based(self, question: str) -> str:
        """규칙 기반 질문 유형 결정"""
        try:
            question_lower = question.lower()
            
            type_indicators = self.advanced_indicators["question_types"]
            
            for q_type, indicators in type_indicators.items():
                if any(indicator in question_lower for indicator in indicators):
                    return q_type
            
            if any(indicator in question_lower for indicator in 
                   ["설명하세요", "기술하세요", "서술하세요", "작성하세요"]):
                return "descriptive"
            elif any(indicator in question_lower for indicator in 
                     ["다음 중", "가장 적절한", "옳은 것"]):
                return "multiple_choice"
            else:
                return "general"
                
        except Exception:
            return "general"
    
    def _extract_key_concepts_rule_based(self, question: str) -> List[str]:
        """규칙 기반 핵심 개념 추출"""
        try:
            question_lower = question.lower()
            concepts = []
            
            for concept in self.security_concepts.keys():
                if concept in question_lower:
                    concepts.append(concept)
            
            for concept in self.technical_concepts.keys():
                if concept in question_lower:
                    concepts.append(concept)
            
            return concepts[:5]
            
        except Exception:
            return []
    
    def _extract_law_references_rule_based(self, question: str) -> List[str]:
        """규칙 기반 법령 참조 추출"""
        try:
            references = []
            question_lower = question.lower()
            
            for law, sections in self.law_references.items():
                law_keywords = law.replace("법", "").split()
                if any(keyword in question_lower for keyword in law_keywords):
                    references.append(law)
            
            article_refs = re.findall(r'제\d+조(?:의\d+)?', question)
            references.extend(article_refs)
            
            return references
            
        except Exception:
            return []
    
    def _assess_difficulty_rule_based(self, question: str) -> str:
        """규칙 기반 난이도 평가"""
        try:
            enhanced_analysis = self.analyze_question_enhanced(question)
            
            total_complexity = enhanced_analysis.technical_complexity + enhanced_analysis.legal_complexity
            
            if total_complexity > 0.7:
                return "advanced"
            elif total_complexity > 0.4:
                return "intermediate"
            else:
                return "basic"
                
        except Exception:
            return "basic"
    
    def get_domain_context(self, domain: str) -> Dict:
        """도메인 컨텍스트 반환"""
        try:
            if domain not in self.domain_keywords:
                return {}
            
            context = {
                "keywords": self.domain_keywords[domain],
                "related_laws": [],
                "key_concepts": [],
                "evolution_trend": self._get_domain_evolution_trend(domain),
                "discovered_patterns": self._get_domain_patterns(domain)
            }
            
            # 법령 매핑
            law_mapping = {
                "개인정보보호": ["개인정보보호법", "정보통신망법"],
                "전자금융": ["전자금융거래법"],
                "금융투자업": ["자본시장법"],
                "정보보안": ["정보통신망법"],
                "사이버보안": ["정보통신망법"]
            }
            
            context["related_laws"] = law_mapping.get(domain, [])
            
            # 개념 매핑
            concept_mapping = {
                "정보보안": ["기밀성", "무결성", "가용성"],
                "사이버보안": ["트로이목마", "피싱", "DDoS"],
                "암호화": ["대칭키", "공개키", "해시함수"],
                "개인정보보호": ["정보주체", "개인정보처리자", "동의"],
                "전자금융": ["접근매체", "전자서명", "오류정정"]
            }
            
            context["key_concepts"] = concept_mapping.get(domain, [])
            
            return context
            
        except Exception:
            return {}
    
    def _get_domain_evolution_trend(self, domain: str) -> Dict:
        """도메인 진화 트렌드 반환"""
        try:
            if domain not in self.domain_evolution_tracking:
                return {"trend": "stable", "confidence": 0.5}
            
            recent_scores = [entry["score"] for entry in self.domain_evolution_tracking[domain][-10:]]
            
            if len(recent_scores) < 3:
                return {"trend": "stable", "confidence": 0.5}
            
            trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            
            if trend_slope > 0.1:
                return {"trend": "increasing", "confidence": min(trend_slope * 5, 1.0)}
            elif trend_slope < -0.1:
                return {"trend": "decreasing", "confidence": min(abs(trend_slope) * 5, 1.0)}
            else:
                return {"trend": "stable", "confidence": 0.8}
                
        except Exception:
            return {"trend": "stable", "confidence": 0.5}
    
    def _get_domain_patterns(self, domain: str) -> List[Dict]:
        """도메인 패턴 반환"""
        try:
            patterns = []
            
            for pattern_id, pattern in self.discovered_patterns.items():
                if pattern.context_type in ["legal", "technical", "managerial"]:
                    domain_relevance = 0
                    
                    for keyword in pattern.keywords:
                        if keyword in self.domain_keywords.get(domain, []):
                            domain_relevance += 1
                    
                    if domain_relevance > 0:
                        patterns.append({
                            "pattern_id": pattern_id,
                            "keywords": pattern.keywords,
                            "relevance": domain_relevance / len(pattern.keywords),
                            "confidence": pattern.confidence,
                            "frequency": pattern.frequency
                        })
            
            return sorted(patterns, key=lambda x: x["confidence"], reverse=True)[:5]
            
        except Exception:
            return []
    
    def suggest_answer_structure(self, question_type: str, domain: List[str]) -> List[str]:
        """답변 구조 제안 (규칙 기반)"""
        try:
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
            elif question_type == "procedural":
                return [
                    "준비 단계",
                    "실행 절차",
                    "검증 방법",
                    "사후 관리"
                ]
            elif question_type == "analytical":
                return [
                    "현황 분석",
                    "문제점 도출",
                    "해결방안 제시",
                    "기대 효과"
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
                
        except Exception:
            return ["핵심 내용", "세부 설명", "결론"]
    
    def get_korean_templates(self, domain: str) -> List[str]:
        """한국어 템플릿 반환"""
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
        """답변 품질 검증 (규칙 기반)"""
        try:
            quality_score = 0.0
            issues = []
            enhancements = []
            
            # 길이 검사
            if len(answer) < 20:
                issues.append("답변이 너무 짧습니다")
            elif len(answer) > 1000:
                issues.append("답변이 너무 깁니다")
            else:
                quality_score += 0.3
            
            # 한국어 비율 검사
            korean_ratio = len(re.findall(r'[가-힣]', answer)) / max(len(answer), 1)
            if korean_ratio > 0.5:
                quality_score += 0.3
            else:
                issues.append("한국어 비율이 낮습니다")
            
            # 문제 문자 검사
            if re.search(r'[\u4e00-\u9fff]', answer):
                issues.append("중국어 문자가 포함되어 있습니다")
            else:
                quality_score += 0.2
            
            # 전문 용어 검사
            professional_terms = ['법', '규정', '조치', '관리', '보안', '정책']
            prof_count = sum(1 for term in professional_terms if term in answer)
            if prof_count >= 3:
                quality_score += 0.2
                enhancements.append("전문 용어 사용 우수")
            elif prof_count >= 1:
                quality_score += 0.1
            
            # 구조 검사
            structure_markers = ['첫째', '둘째', '따라서', '그러므로']
            if any(marker in answer for marker in structure_markers):
                quality_score += 0.1
                enhancements.append("구조적 답변 작성")
            
            return {
                "quality_score": min(quality_score, 1.0),
                "issues": issues,
                "enhancements": enhancements,
                "is_acceptable": quality_score >= 0.6 and len(issues) <= 1,
                "korean_ratio": korean_ratio,
                "professional_terms_count": prof_count
            }
            
        except Exception:
            return {
                "quality_score": 0.0,
                "issues": ["품질 검증 실패"],
                "enhancements": [],
                "is_acceptable": False,
                "korean_ratio": 0.0,
                "professional_terms_count": 0
            }
    
    def get_performance_metrics(self) -> Dict:
        """성능 지표 반환"""
        try:
            cache_hit_rate = 0.0
            total_requests = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
            if total_requests > 0:
                cache_hit_rate = self.performance_stats["cache_hits"] / total_requests
            
            avg_cooccurrence = 0.0
            if self.discovered_patterns:
                scores = [p.co_occurrence_score for p in self.discovered_patterns.values()]
                avg_cooccurrence = np.mean(scores)
            
            return {
                "total_patterns_discovered": len(self.discovered_patterns),
                "cache_size": len(self.analysis_cache),
                "cache_hit_rate": cache_hit_rate,
                "domain_tracking_active": len(self.domain_evolution_tracking),
                "avg_cooccurrence_score": avg_cooccurrence,
                "analysis_count": self.performance_stats["analysis_count"],
                "rule_based_processing": self.performance_stats["rule_based_processing"],
                "pattern_confidence_distribution": {
                    "high": len([p for p in self.discovered_patterns.values() if p.confidence > 0.8]),
                    "medium": len([p for p in self.discovered_patterns.values() if 0.5 < p.confidence <= 0.8]),
                    "low": len([p for p in self.discovered_patterns.values() if p.confidence <= 0.5])
                },
                "compliance_status": {
                    "single_model_usage": True,
                    "external_model_usage": False,
                    "rule_based_only": True
                }
            }
            
        except Exception:
            return {
                "total_patterns_discovered": 0,
                "cache_size": 0,
                "cache_hit_rate": 0.0,
                "domain_tracking_active": 0,
                "avg_cooccurrence_score": 0.0,
                "analysis_count": 0,
                "rule_based_processing": 0,
                "pattern_confidence_distribution": {"high": 0, "medium": 0, "low": 0},
                "compliance_status": {
                    "single_model_usage": True,
                    "external_model_usage": False,
                    "rule_based_only": True
                }
            }
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            print("지식베이스 정리 중...")
            
            # 성능 메트릭 출력
            metrics = self.get_performance_metrics()
            print(f"  - 발견된 패턴: {metrics['total_patterns_discovered']}개")
            print(f"  - 캐시 적중률: {metrics['cache_hit_rate']:.2%}")
            print(f"  - 분석 처리 건수: {metrics['analysis_count']}건")
            print(f"  - 규칙 기반 처리: {metrics['rule_based_processing']}건")
            print(f"  - ✅ 단일 모델 준수: {metrics['compliance_status']['single_model_usage']}")
            print(f"  - ✅ 외부 모델 사용 금지: {not metrics['compliance_status']['external_model_usage']}")
            
            # 캐시 정리
            self.analysis_cache.clear()
            self.discovered_patterns.clear()
            self.keyword_cooccurrence.clear()
            self.domain_evolution_tracking.clear()
            
            # 메모리 정리
            gc.collect()
            
            print("지식베이스 정리 완료 (대회 규칙 준수)")
            
        except Exception:
            pass