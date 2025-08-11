# knowledge_base.py

import re
import itertools
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

class FinancialSecurityKnowledgeBase:
    
    def __init__(self):
        self.domain_keywords = self._initialize_domain_keywords()
        self.law_references = self._initialize_law_references()
        self.security_concepts = self._initialize_security_concepts()
        self.financial_concepts = self._initialize_financial_concepts()
        self.technical_concepts = self._initialize_technical_concepts()
        
        self.word_patterns = self._build_word_patterns()
        self.ngram_patterns = self._build_ngram_patterns()
        self.context_patterns = self._build_context_patterns()
        self.answer_associations = self._build_answer_associations()
        
        self.pattern_cache = {}
        self.learned_combinations = defaultdict(Counter)
        
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        return {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보처리자", "개인정보보호법", "개인정보처리",
                "수집", "이용", "제공", "파기", "동의", "열람", "정정", "삭제",
                "개인정보보호위원회", "개인정보보호책임자", "개인정보취급자",
                "민감정보", "고유식별정보", "안전성확보조치", "영향평가",
                "개인정보처리방침", "개인정보파일", "개인정보이용내역통지",
                "가명정보", "익명정보", "결합전문기관", "개인정보침해신고센터",
                "개인정보보호인증", "개인정보보호교육", "개인정보보호감사",
                "개인정보이동권", "프로파일링", "자동화된결정", "개인정보보호담당자"
            ],
            "전자금융": [
                "전자금융거래", "전자금융거래법", "전자적장치", "접근매체", "전자서명",
                "전자인증", "금융기관", "전자금융업", "전자지급수단", "전자화폐",
                "전자금융거래기록", "오류정정", "손해배상", "약관", "이용자",
                "전자금융감독규정", "전자금융거래분쟁", "전자금융업감독규정",
                "전자금융보안", "전자금융사고", "전자금융통계", "전자금융위원회",
                "전자금융거래내역", "전자금융거래확인", "전자금융거래취소", "전자금융거래정지"
            ],
            "정보보안": [
                "정보보안", "정보보호", "정보보안관리체계", "ISMS", "ISMS-P",
                "보안정책", "보안통제", "위험관리", "취약점", "보안사고",
                "접근통제", "암호화", "네트워크보안", "시스템보안", "데이터보안",
                "보안감사", "보안교육", "보안인식", "보안관제", "침해대응",
                "보안위험평가", "보안통제평가", "보안관리체계인증", "보안사고대응",
                "보안모니터링", "보안점검", "보안컨설팅", "보안솔루션", "보안기술"
            ],
            "암호화": [
                "암호화", "복호화", "암호", "암호키", "대칭키", "공개키", "개인키",
                "해시함수", "전자서명", "인증서", "PKI", "암호알고리즘",
                "AES", "RSA", "SHA", "MD5", "DES", "3DES", "ECC",
                "키관리", "키분배", "키교환", "키생성", "키폐기",
                "키유도", "키래핑", "키스케줄링", "키확장", "키복구",
                "키에스크로", "키백업", "키순환", "키길이", "키강도"
            ],
            "사이버보안": [
                "사이버보안", "사이버공격", "해킹", "악성코드", "멀웨어", "바이러스",
                "웜", "트로이목마", "랜섬웨어", "스파이웨어", "애드웨어", "루트킷",
                "피싱", "스미싱", "파밍", "스피어피싱", "사회공학", "제로데이",
                "APT", "DDoS", "DDOS", "봇넷", "C&C", "백도어", "키로거",
                "SQL인젝션", "XSS", "CSRF", "버퍼오버플로우", "힙오버플로우",
                "포맷스트링", "정수오버플로우", "경쟁조건", "시간차공격", "사이드채널공격"
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험분석", "위험식별", "위험측정", "위험통제",
                "위험모니터링", "위험보고", "위험수용", "위험회피", "위험전가", "위험완화",
                "위험관리체계", "위험관리정책", "위험관리조직", "위험관리절차",
                "정보보호위험", "운영위험", "시스템위험", "재무위험",
                "위험관리위원회", "위험관리담당자", "위험관리계획", "위험관리교육"
            ],
            "관리체계": [
                "관리체계", "정보보호관리체계", "정보보안관리체계", "PDCA", "Plan", "Do", "Check", "Act",
                "정책", "조직", "자산관리", "인적보안", "물리보안", "시스템보안",
                "네트워크보안", "접근통제", "시스템개발", "공급업체관리", "사고관리",
                "업무연속성", "준수성", "내부감사", "경영검토",
                "관리체계인증", "관리체계평가", "관리체계교육", "관리체계컨설팅"
            ],
            "재해복구": [
                "재해복구", "재해복구계획", "BCP", "업무연속성", "재해복구센터", "DRP",
                "재해복구시스템", "재해복구시나리오", "재해복구훈련", "재해복구테스트",
                "백업", "복구", "RTO", "RPO", "복구목표시간", "복구목표시점",
                "핫사이트", "콜드사이트", "웜사이트", "클러스터링", "미러링",
                "재해복구절차", "재해복구매뉴얼", "재해복구조직", "재해복구예산"
            ],
            "금융투자업": [
                "금융투자업", "투자매매업", "투자중개업", "투자자문업", "투자일임업",
                "집합투자업", "신탁업", "소비자금융업", "보험중개업", "자본시장법",
                "금융투자상품", "증권", "파생상품", "집합투자증권", "투자자보호",
                "금융투자회사", "금융투자협회", "금융투자위원회", "금융투자감독"
            ],
            "접근제어": [
                "접근제어", "접근권한", "인증", "인가", "식별", "계정관리",
                "패스워드", "다중인증", "이중인증", "생체인증", "토큰인증",
                "싱글사인온", "SSO", "LDAP", "Active Directory", "RBAC", "MAC", "DAC",
                "접근제어정책", "접근제어모델", "접근제어시스템", "접근제어감사"
            ],
            "네트워크보안": [
                "네트워크보안", "방화벽", "침입탐지시스템", "침입방지시스템", "IDS", "IPS",
                "가상사설망", "VPN", "네트워크분할", "네트워크모니터링", "트래픽분석",
                "프록시", "NAT", "VLAN", "DMZ", "허니팟", "네트워크접근제어", "NAC",
                "네트워크보안정책", "네트워크보안아키텍처", "네트워크보안감사"
            ]
        }
    
    def _build_word_patterns(self) -> Dict[str, Dict]:
        patterns = {}
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if len(keyword) >= 2:
                    patterns[keyword] = {
                        "domain": domain,
                        "weight": len(keyword) * 0.1,
                        "frequency": 1
                    }
        
        combination_patterns = {}
        for domain, keywords in self.domain_keywords.items():
            for i, word1 in enumerate(keywords[:20]):
                for word2 in keywords[i+1:min(i+11, len(keywords))]:
                    combo = f"{word1} {word2}"
                    combination_patterns[combo] = {
                        "domain": domain,
                        "weight": (len(word1) + len(word2)) * 0.15,
                        "frequency": 1
                    }
        
        patterns.update(combination_patterns)
        return patterns
    
    def _build_ngram_patterns(self) -> Dict[str, Dict]:
        ngrams = {}
        
        common_phrases = [
            "해당하지 않는", "적절하지 않은", "옳지 않은", "틀린 것",
            "가장 적절한", "가장 중요한", "반드시 포함", "우선적으로 고려",
            "법령에 따라", "규정에 의해", "정책을 수립", "체계를 구축",
            "방안을 마련", "조치를 취해", "절차를 수행", "관리를 위해",
            "보안을 강화", "위험을 평가", "대책을 마련", "시스템을 운영",
            "교육을 실시", "점검을 수행", "감사를 실행", "모니터링을 통해",
            "개선을 도모", "효과를 제고", "품질을 향상", "성능을 최적화"
        ]
        
        for phrase in common_phrases:
            words = phrase.split()
            for i in range(len(words)):
                for j in range(i+1, min(i+4, len(words)+1)):
                    ngram = " ".join(words[i:j])
                    if len(ngram) >= 3:
                        ngrams[ngram] = {
                            "type": "phrase",
                            "weight": len(words[i:j]) * 0.2,
                            "context": phrase
                        }
        
        question_patterns = [
            "정의로 가장", "의미로 적절한", "특징을 설명",
            "방법을 기술", "절차를 서술", "과정을 논술",
            "원인을 분석", "결과를 예측", "영향을 평가",
            "대안을 제시", "방안을 도출", "전략을 수립"
        ]
        
        for pattern in question_patterns:
            ngrams[pattern] = {
                "type": "question_pattern",
                "weight": 0.3,
                "context": "질문_유형"
            }
        
        return ngrams
    
    def _build_context_patterns(self) -> Dict[str, Dict]:
        contexts = {}
        
        negative_patterns = [
            "해당하지", "적절하지", "옳지", "맞지", "관련없는",
            "무관한", "제외한", "빼고", "아닌", "틀린"
        ]
        
        for pattern in negative_patterns:
            contexts[pattern] = {
                "type": "negative",
                "weight": 0.4,
                "answer_bias": {"1": 0.3, "3": 0.25, "5": 0.2, "2": 0.15, "4": 0.1}
            }
        
        positive_patterns = [
            "가장 적절한", "가장 중요한", "올바른", "정확한",
            "필수적인", "우선적인", "기본적인", "핵심적인"
        ]
        
        for pattern in positive_patterns:
            contexts[pattern] = {
                "type": "positive",
                "weight": 0.35,
                "answer_bias": {"2": 0.3, "1": 0.25, "3": 0.2, "4": 0.15, "5": 0.1}
            }
        
        definition_patterns = [
            "정의", "의미", "개념", "용어", "뜻"
        ]
        
        for pattern in definition_patterns:
            contexts[pattern] = {
                "type": "definition",
                "weight": 0.3,
                "answer_bias": {"2": 0.4, "1": 0.25, "3": 0.2, "4": 0.1, "5": 0.05}
            }
        
        return contexts
    
    def _build_answer_associations(self) -> Dict[str, Dict]:
        associations = {
            "금융투자업": {
                "소비자금융업": {"answer": "1", "confidence": 0.9},
                "보험중개업": {"answer": "5", "confidence": 0.85},
                "투자매매업": {"answer": "2", "confidence": 0.8},
                "투자중개업": {"answer": "3", "confidence": 0.8},
                "투자자문업": {"answer": "4", "confidence": 0.8}
            },
            "위험관리": {
                "위험수용": {"answer": "2", "confidence": 0.9},
                "위험회피": {"answer": "1", "confidence": 0.8},
                "위험전가": {"answer": "3", "confidence": 0.8},
                "위험완화": {"answer": "4", "confidence": 0.8}
            },
            "관리체계": {
                "경영진 참여": {"answer": "2", "confidence": 0.9},
                "정책 수립": {"answer": "1", "confidence": 0.8},
                "자원 할당": {"answer": "3", "confidence": 0.7},
                "책임자 지정": {"answer": "4", "confidence": 0.7}
            },
            "재해복구": {
                "개인정보파기": {"answer": "3", "confidence": 0.9},
                "복구절차": {"answer": "1", "confidence": 0.8},
                "비상연락": {"answer": "2", "confidence": 0.8},
                "백업계획": {"answer": "4", "confidence": 0.7}
            }
        }
        return associations
    
    def analyze_word_patterns(self, text: str) -> Dict:
        text_lower = text.lower()
        found_patterns = []
        total_weight = 0
        domain_scores = defaultdict(float)
        
        for pattern, info in self.word_patterns.items():
            if pattern in text_lower:
                found_patterns.append({
                    "pattern": pattern,
                    "domain": info["domain"],
                    "weight": info["weight"]
                })
                total_weight += info["weight"]
                domain_scores[info["domain"]] += info["weight"]
        
        for ngram, info in self.ngram_patterns.items():
            if ngram in text_lower:
                found_patterns.append({
                    "pattern": ngram,
                    "type": info["type"],
                    "weight": info["weight"]
                })
                total_weight += info["weight"]
        
        context_analysis = self._analyze_context_patterns(text_lower)
        
        return {
            "patterns": found_patterns,
            "total_weight": total_weight,
            "domain_scores": dict(domain_scores),
            "primary_domain": max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else None,
            "context": context_analysis
        }
    
    def _analyze_context_patterns(self, text: str) -> Dict:
        context_results = {
            "type": "neutral",
            "confidence": 0.5,
            "answer_bias": {}
        }
        
        for pattern, info in self.context_patterns.items():
            if pattern in text:
                context_results = {
                    "type": info["type"],
                    "confidence": info["weight"],
                    "answer_bias": info["answer_bias"]
                }
                break
        
        return context_results
    
    def get_answer_suggestion(self, text: str, context: Dict = None) -> Tuple[Optional[str], float]:
        analysis = self.analyze_word_patterns(text)
        
        for domain, associations in self.answer_associations.items():
            if domain in analysis["primary_domain"] if analysis["primary_domain"] else "":
                for keyword, answer_info in associations.items():
                    if keyword in text.lower():
                        return answer_info["answer"], answer_info["confidence"]
        
        if analysis["context"]["answer_bias"]:
            best_answer = max(analysis["context"]["answer_bias"].items(), key=lambda x: x[1])
            return best_answer[0], analysis["context"]["confidence"]
        
        return None, 0.0
    
    def learn_pattern(self, text: str, answer: str, confidence: float):
        if confidence < 0.4:
            return
        
        words = re.findall(r'[가-힣]{2,}', text.lower())
        
        for word in words[:10]:
            if word not in self.word_patterns:
                self.word_patterns[word] = {
                    "domain": "학습됨",
                    "weight": confidence * 0.1,
                    "frequency": 1
                }
            else:
                self.word_patterns[word]["frequency"] += 1
                self.word_patterns[word]["weight"] += confidence * 0.05
        
        for i in range(len(words)-1):
            combo = f"{words[i]} {words[i+1]}"
            self.learned_combinations[combo][answer] += confidence
        
        for i in range(len(words)-2):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(trigram) <= 30:
                self.learned_combinations[trigram][answer] += confidence * 1.2
    
    def get_learned_suggestion(self, text: str) -> Tuple[Optional[str], float]:
        text_lower = text.lower()
        suggestions = defaultdict(float)
        
        for pattern, answer_counts in self.learned_combinations.items():
            if pattern in text_lower:
                total_count = sum(answer_counts.values())
                if total_count >= 2:
                    for answer, count in answer_counts.items():
                        weight = (count / total_count) * min(total_count * 0.1, 0.8)
                        suggestions[answer] += weight
        
        if suggestions:
            best_answer = max(suggestions.items(), key=lambda x: x[1])
            return best_answer[0], min(best_answer[1], 0.9)
        
        return None, 0.0
    
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
            },
            "신용정보법": {
                "신용정보보호": "신용정보의 이용 및 보호에 관한 법률",
                "신용정보처리": "신용정보법 제15조",
                "신용정보제공": "신용정보법 제17조"
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
    
    def _initialize_financial_concepts(self) -> Dict[str, str]:
        return {
            "금융투자업": "금융투자상품에 관하여 투자매매업, 투자중개업, 투자자문업, 투자일임업, 집합투자업 또는 신탁업을 하는 것",
            "투자매매업": "금융투자상품을 매도 또는 매수하는 업",
            "투자중개업": "투자자로부터 금융투자상품의 매매, 그 밖의 거래에 관한 주문을 받아 금융투자상품의 매매, 그 밖의 거래를 중개, 주선 또는 대리하는 업",
            "투자자문업": "금융투자상품의 가치 또는 금융투자상품에 대한 투자판단에 관하여 자문하는 업",
            "투자일임업": "투자자로부터 금융투자상품에 대한 투자판단의 전부 또는 일부를 일임받아 투자자별로 구분하여 자산을 투자, 운용하는 업",
            "소비자금융업": "일반 소비자를 대상으로 하는 소액 신용대출, 신용카드 발행 등의 업무",
            "보험중개업": "보험계약의 체결을 중개하는 업무"
        }
    
    def _initialize_technical_concepts(self) -> Dict[str, str]:
        return {
            "트로이목마": "정상적인 프로그램으로 위장하여 시스템에 침입한 후 악의적인 행위를 수행하는 악성코드",
            "RAT": "원격 접근 트로이목마(Remote Access Trojan)로, 공격자가 감염된 시스템을 원격으로 제어할 수 있게 하는 악성코드",
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
            "difficulty_level": self._assess_difficulty(question),
            "word_patterns": self.analyze_word_patterns(question)
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
        
        technical_terms = ["암호화", "해시", "PKI", "SSL", "TLS", "VPN", "IDS", "IPS", "SIEM"]
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