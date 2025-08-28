# data_processor.py

import re
import unicodedata
from typing import Dict, List, Tuple
from config import KOREAN_REQUIREMENTS, POSITIONAL_ANALYSIS


class DataProcessor:
    """데이터 처리"""

    def __init__(self):
        self._initialize_data()
        self.korean_requirements = KOREAN_REQUIREMENTS.copy()
        self.domain_keywords = self._setup_domain_keywords()
        self.positional_config = POSITIONAL_ANALYSIS.copy()

    def _initialize_data(self):
        """데이터 초기화"""
        
        # 객관식 패턴
        self.mc_patterns = [
            r"1\s+[가-힣\w].*\n2\s+[가-힣\w].*\n3\s+[가-힣\w].*\n4\s+[가-힣\w].*\n5\s+[가-힣\w]",
            r"1\s+[가-힣\w].*2\s+[가-힣\w].*3\s+[가-힣\w].*4\s+[가-힣\w].*5\s+[가-힣\w]",
            r"1\)\s*[가-힣\w].*2\)\s*[가-힣\w].*3\)\s*[가-힣\w].*4\)\s*[가-힣\w].*5\)\s*[가-힣\w]",
            r"1\.\s*[가-힣\w].*2\.\s*[가-힣\w].*3\.\s*[가-힣\w].*4\.\s*[가-힣\w].*5\.\s*[가-힣\w]",
            r"①.*②.*③.*④.*⑤",
            r"\n1\s+.*\n2\s+.*\n3\s+.*\n4\s+.*\n5\s+"
        ]

        # 주관식 패턴
        self.subj_patterns = [
            r"설명하세요$", r"기술하세요$", r"서술하세요$", r"작성하세요$",
            r".*설명하세요\.$", r".*기술하세요\.$", r".*서술하세요\.$",
            r".*을\s*설명하세요", r".*를\s*설명하세요", r".*을\s*기술하세요", r".*를\s*기술하세요",
            r".*방안을\s*기술하세요", r".*절차를\s*설명하세요", r".*특징.*설명하세요",
            r".*지표.*설명하세요", r".*기관.*기술하세요", r".*대응.*방안.*기술하세요"
        ]

        # 객관식 키워드
        self.mc_keywords = [
            "해당하지.*않는.*것", "적절하지.*않는.*것", "옳지.*않는.*것", "틀린.*것",
            "맞는.*것", "옳은.*것", "적절한.*것", "올바른.*것", "가장.*적절한.*것",
            "가장.*옳은.*것", "구분.*해당하지.*않는.*것", "다음.*중.*것은", "다음.*중.*것",
            "다음.*보기.*중", "무엇인가\\?$", "어떤.*것인가\\?$", "몇.*개인가\\?$"
        ]

        # 질문 의도 패턴
        self.question_intent_patterns = {
            "기관_묻기": [
                "기관.*기술하세요", "기관.*설명하세요", "기관.*서술하세요", "기관.*무엇",
                "어떤.*기관", "어느.*기관", "기관.*어디", "분쟁조정.*신청.*기관",
                "조정.*신청.*기관", "분쟁.*조정.*기관", "신청.*수.*있는.*기관"
            ],
            "특징_묻기": [
                "특징.*설명하세요", "특징.*기술하세요", "특징.*서술하세요", "어떤.*특징",
                "주요.*특징", "특징.*무엇", "성격.*설명", "성질.*설명", "속성.*설명",
                "트로이.*특징", "RAT.*특징", "원격제어.*특징", "딥페이크.*특징"
            ],
            "지표_묻기": [
                "지표.*설명하세요", "탐지.*지표", "주요.*지표", "어떤.*지표", "지표.*무엇",
                "징후.*설명", "신호.*설명", "패턴.*설명", "행동.*패턴", "활동.*패턴",
                "탐지.*지표.*설명하세요", "주요.*탐지.*지표"
            ],
            "방안_묻기": [
                "방안.*기술하세요", "방안.*설명하세요", "대응.*방안", "해결.*방안",
                "관리.*방안", "어떤.*방안", "대책.*설명", "조치.*방안", "처리.*방안",
                "딥페이크.*대응.*방안", "금융권.*대응.*방안"
            ],
            "절차_묻기": [
                "절차.*설명하세요", "절차.*기술하세요", "어떤.*절차", "처리.*절차",
                "진행.*절차", "수행.*절차", "실행.*절차", "과정.*설명", "단계.*설명",
                "동의.*절차", "신고.*절차"
            ],
            "비율_묻기": [
                "비율.*얼마", "기준.*비율", "비율.*무엇", "몇.*퍼센트", "어느.*정도",
                "기준.*얼마", "정보기술부문.*비율", "예산.*비율", "인력.*비율"
            ]
        }

        self._setup_korean_recovery_mappings()

    def _setup_domain_keywords(self):
        """도메인 키워드 설정"""
        return {
            "개인정보보호": [
                "개인정보", "정보주체", "개인정보보호법", "민감정보", "고유식별정보",
                "수집", "이용", "제공", "파기", "동의", "법정대리인", "아동", "처리",
                "개인정보보호위원회", "개인정보영향평가", "개인정보관리체계",
                "만 14세", "미만 아동", "중요한 요소", "경영진", "접근권한",
                "개인정보침해신고센터", "정보주체", "개인정보처리방침",
            ],
            "전자금융": [
                "전자금융", "전자적", "접근매체", "전자금융거래법", "전자서명",
                "전자인증", "분쟁조정", "전자지급수단", "금융감독원", "한국은행",
                "전자금융업", "전자금융분쟁조정위원회", "전자금융거래", "이용자",
                "금융통화위원회", "자료제출", "통화신용정책", "지급결제제도",
                "정보기술부문", "비율", "예산", "인력", "전자금융감독규정",
                "전자금융업자", "전자금융업무", "전자금융서비스",
            ],
            "사이버보안": [
                "트로이", "악성코드", "멀웨어", "바이러스", "피싱", "스미싱", "랜섬웨어",
                "해킹", "딥페이크", "원격제어", "RAT", "원격접근", "봇넷", "백도어",
                "SBOM", "소프트웨어 구성 요소", "원격제어 악성코드", "탐지 지표",
                "보안 위협", "특징", "주요 탐지", "금융권", "활용", "이유",
                "디지털 지갑", "보안 위협", "딥보이스", "탐지 기술", "사이버공격",
                "악성 프로그램", "보안 사고", "침해 사고",
            ],
            "정보보안": [
                "정보보안", "보안관리", "ISMS", "보안정책", "접근통제", "암호화",
                "방화벽", "침입탐지", "침입방지시스템", "IDS", "IPS", "보안관제",
                "로그관리", "백업", "복구", "재해복구", "BCP", "정보보안관리체계",
                "정보보호", "관리체계", "복구 절차", "비상연락체계", "개인정보 파기",
                "복구 목표시간", "3대 요소", "SMTP", "프로토콜", "보안상 주요 역할",
                "취약점 스캐닝", "보안 감사", "보안 교육", "정보보호최고책임자",
            ],
            "금융투자": [
                "금융투자업", "투자자문업", "투자매매업", "투자중개업", "소비자금융업",
                "보험중개업", "자본시장법", "집합투자업", "신탁업", "펀드", "파생상품",
                "투자자보호", "적합성원칙", "설명의무", "금융산업", "구분", "해당하지 않는",
                "투자일임업", "신용정보회사", "신용회복",
            ],
            "위험관리": [
                "위험관리", "위험평가", "위험대응", "위험수용", "리스크", "내부통제",
                "컴플라이언스", "위험식별", "위험분석", "위험모니터링", "위험회피",
                "위험 관리 계획", "수행인력", "위험 대응 전략", "적절하지 않은",
                "위험통제", "리스크 관리",
            ],
            "정보통신": [
                "정보통신시설", "집적된 정보통신시설", "정보통신서비스", 
                "과학기술정보통신부장관", "보고", "중단", "발생", "일시", "장소", 
                "원인", "법적 책임", "피해내용", "응급조치", "정보통신기반 보호법",
                "SPF", "Sender Policy Framework", "프로토콜", "네트워크",
                "키 분배", "대칭 키", "국내대리인",
            ],
            "기타": [
                "청문", "절차", "제출", "기준", "관리", "통제", "운영", "업무", "법", "조",
                "규정", "시행", "적용", "준수", "이행", "수행", "담당", "책임", "의무",
                "권한", "허가", "승인", "신고", "등록", "지정", "선정", "검토", "평가",
                "감독", "감시", "점검", "조사", "확인", "검증", "분석", "판단", "결정",
            ]
        }

    def _setup_korean_recovery_mappings(self):
        """한국어 복구 매핑 설정"""
        self.korean_recovery_mapping = {
            # 깨진 유니코드 문자
            "\\u1100": "", "\\u1101": "", "\\u1102": "", "\\u1103": "", "\\u1104": "",
            "\\u1105": "", "\\u1106": "", "\\u1107": "", "\\u1108": "", "\\u1109": "",
            # 공백이 들어간 한국어 수정
            "작 로": "으로", "렴": "련", "니 터": "니터", "지 속": "지속", "모 니": "모니",
            "체 계": "체계", "관 리": "관리", "법 령": "법령", "규 정": "규정", "조 치": "조치",
            "절 차": "절차", "대 응": "대응", "방 안": "방안", "기 관": "기관"
        }

    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출"""
        
        # 1단계: 명확한 주관식 패턴 확인
        for pattern in self.subj_patterns:
            try:
                if re.search(pattern, question, re.IGNORECASE):
                    return "subjective", 0
            except Exception:
                continue
        
        # 2단계: 명확한 객관식 패턴 확인
        for pattern in self.mc_patterns:
            try:
                if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                    return self._extract_mc_choice_count(question)
            except Exception:
                continue

        # 3단계: 선택지 개수 기반 판단
        choice_count = self._count_valid_choices(question)
        if choice_count >= 4:
            return "multiple_choice", choice_count

        # 4단계: 키워드 기반 분석
        question_type_score = self._calculate_question_type_score(question)
        if question_type_score > 2.0:
            return "multiple_choice", max(choice_count, 5)
        elif question_type_score < -1.0:
            return "subjective", 0

        # 5단계: 최종 판단
        return self._final_type_determination(question, choice_count, question_type_score)

    def _calculate_question_type_score(self, question: str) -> float:
        """질문 유형 점수 계산"""
        try:
            question_lower = question.lower()
            score = 0.0
            
            # 객관식 패턴
            for pattern in self.mc_keywords:
                if re.search(pattern, question_lower):
                    score += 2.0
            
            # 주관식 패턴
            subj_patterns = ["설명하세요", "기술하세요", "서술하세요", "방안을.*기술하세요"]
            for pattern in subj_patterns:
                if pattern in question_lower:
                    score -= 2.0
            
            return score
        except Exception:
            return 0.0

    def _extract_mc_choice_count(self, question: str) -> Tuple[str, int]:
        """객관식 선택지 개수 추출"""
        lines = question.split("\n")
        choice_numbers = []
        
        patterns = [
            r"^(\d+)\s+(.+)",
            r"^(\d+)\)\s*(.+)",
            r"^(\d+)\.\s*(.+)",
            r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*(.+)"
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                try:
                    if pattern.startswith("^[①②③④⑤"):
                        if re.match(pattern, line):
                            choice_numbers.append(len(choice_numbers) + 1)
                            break
                    else:
                        match = re.match(pattern, line)
                        if match:
                            num = int(match.group(1))
                            content = match.group(2).strip()
                            if 1 <= num <= 10 and len(content) >= 2:
                                choice_numbers.append(num)
                            break
                except (ValueError, IndexError):
                    continue

        if choice_numbers:
            max_choice = max(choice_numbers)
            if max_choice <= 10:
                return "multiple_choice", max_choice

        return "multiple_choice", 5

    def _count_valid_choices(self, question: str) -> int:
        """유효한 선택지 개수 계산"""
        lines = question.split("\n")
        valid_choices = 0
        found_numbers = set()
        
        patterns = [
            r"^(\d+)\s+[가-힣\w]{2,}",
            r"^(\d+)\)\s*[가-힣\w]{2,}",
            r"^(\d+)\.\s*[가-힣\w]{2,}",
            r"^[①②③④⑤⑥⑦⑧⑨⑩]\s*[가-힣\w]{2,}"
        ]

        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
                
            for pattern in patterns:
                try:
                    if pattern.startswith("^[①②③④⑤"):
                        if re.match(pattern, line):
                            valid_choices += 1
                            break
                    else:
                        match = re.match(pattern, line)
                        if match:
                            try:
                                num = int(match.group(1))
                                if 1 <= num <= 10 and num not in found_numbers:
                                    valid_choices += 1
                                    found_numbers.add(num)
                                    break
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    continue

        return valid_choices

    def _final_type_determination(self, question: str, choice_count: int, type_score: float) -> Tuple[str, int]:
        """최종 유형 결정"""
        question_lower = question.lower()
        
        # 명확한 패턴이 있는 경우
        if abs(type_score) > 3.0:
            if type_score > 0:
                return "multiple_choice", max(choice_count, 5)
            else:
                return "subjective", 0
        
        # 선택지가 충분히 있는 경우
        if choice_count >= 4:
            return "multiple_choice", choice_count
        
        # 길이 기반 판단
        if len(question) > 300:
            if any(word in question_lower for word in ["설명", "기술", "서술", "방안", "절차"]):
                return "subjective", 0
        elif len(question) < 200 and choice_count >= 2:
            return "multiple_choice", max(choice_count, 5)
        
        # 기본 결정
        if choice_count >= 2:
            return "multiple_choice", max(choice_count, 5)
        else:
            return "subjective", 0

    def extract_domain(self, question: str, question_number: int = None) -> str:
        """도메인 추출"""
        question_lower = question.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 키워드 길이에 따른 가중치
                    if len(keyword) >= 6:
                        score += 4
                    elif len(keyword) >= 4:
                        score += 3
                    elif len(keyword) >= 2:
                        score += 1
            
            # 위치별 가중치 적용
            if question_number is not None and score > 0:
                position_weight = self._get_position_weight(question_number)
                score *= position_weight
            
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return self._classify_unknown_domain(question_lower, question_number)

        best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # "기타" 도메인인 경우 추가 분석
        if best_domain == "기타":
            enhanced_domain = self._enhance_generic_classification(question_lower)
            if enhanced_domain != "기타":
                best_domain = enhanced_domain
        
        return best_domain

    def _get_position_weight(self, question_number: int) -> float:
        """위치별 가중치 계산"""
        if question_number <= 100:
            return self.positional_config["position_weight_factors"]["early"]
        elif question_number <= 300:
            return self.positional_config["position_weight_factors"]["middle"]
        else:
            return self.positional_config["position_weight_factors"]["late"]

    def _enhance_generic_classification(self, question_lower: str) -> str:
        """기타 분류 정확도 향상"""
        
        # 법령별 도메인 분류
        law_mappings = {
            "개인정보보호법": "개인정보보호",
            "전자금융거래법": "전자금융",
            "정보통신기반": "정보통신",
            "자본시장법": "금융투자",
            "신용정보법": "금융투자",
        }
        
        for law, domain in law_mappings.items():
            if law in question_lower:
                return domain
        
        # 기관별 도메인 분류
        institution_mappings = {
            "개인정보보호위원회": "개인정보보호",
            "금융감독원": "전자금융",
            "한국은행": "전자금융",
            "과학기술정보통신부": "정보통신",
        }
        
        for institution, domain in institution_mappings.items():
            if institution in question_lower:
                return domain
        
        # 기술 용어별 분류
        tech_mappings = {
            "암호화": "정보보안", 
            "해시": "정보보안", 
            "키": "정보보안",
            "취약점": "정보보안", 
            "해킹": "사이버보안", 
            "침입": "사이버보안",
            "투자": "금융투자", 
            "펀드": "금융투자", 
            "주식": "금융투자",
            "청문": "전자금융",
            "spf": "정보통신",
            "프로토콜": "정보통신",
        }
        
        for tech_keyword, domain in tech_mappings.items():
            if tech_keyword in question_lower:
                return domain
        
        return "기타"

    def _classify_unknown_domain(self, question_lower: str, question_number: int = None) -> str:
        """미분류 도메인 분류"""
        
        # 위치 기반 도메인 추정
        if question_number is not None:
            if question_number > 300:
                # 후반부에는 복잡한 문제가 많음
                if any(word in question_lower for word in ["법", "조", "항", "규정"]):
                    return "개인정보보호"
                elif any(word in question_lower for word in ["관리", "통제", "운영"]):
                    return "정보보안"
        
        return self._enhance_generic_classification(question_lower)

    def analyze_question_intent(self, question: str) -> Dict:
        """질문 의도 분석"""
        question_lower = question.lower()

        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형"
        }

        intent_scores = {}

        for intent_type, patterns in self.question_intent_patterns.items():
            score = 0
            matched_patterns = []

            for pattern in patterns:
                try:
                    matches = re.findall(pattern, question, re.IGNORECASE)
                    if matches:
                        score += 2.0
                        matched_patterns.append(pattern)
                except Exception:
                    continue

            if score > 0:
                intent_scores[intent_type] = {"score": score, "patterns": matched_patterns}

        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]

            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 4.0, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]

            intent_analysis.update(self._determine_answer_requirements(best_intent[0], question_lower))
        
        return intent_analysis

    def _determine_answer_requirements(self, intent_type: str, question_lower: str) -> Dict:
        """답변 요구사항 결정"""
        
        requirements = {}
        
        if "기관" in intent_type:
            requirements.update({
                "answer_type_required": "기관명"
            })
        elif "특징" in intent_type:
            requirements.update({
                "answer_type_required": "특징설명"
            })
        elif "지표" in intent_type:
            requirements.update({
                "answer_type_required": "지표나열"
            })
        elif "방안" in intent_type:
            requirements.update({
                "answer_type_required": "방안제시"
            })
        elif "비율" in intent_type:
            requirements.update({
                "answer_type_required": "수치설명"
            })
        
        return requirements

    def analyze_question_difficulty(self, question: str, question_number: int = None) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()

        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", "트로이", "멀웨어",
            "랜섬웨어", "딥페이크", "피싱", "접근매체", "전자서명", "rat",
            "개인정보보호법", "자본시장법", "전자금융거래법", "spf"
        ]

        difficulty_score = 0
        
        term_count = sum(1 for term in technical_terms if term in question_lower)
        if term_count >= 3:
            difficulty_score += 3
        elif term_count >= 1:
            difficulty_score += 1
            
        length = len(question)
        if length > 400:
            difficulty_score += 2
        elif length > 200:
            difficulty_score += 1

        if any(law in question_lower for law in ["법", "조", "항", "규정", "지침"]):
            difficulty_score += 1

        # 위치별 난이도 조정
        if question_number is not None:
            if question_number > 300:
                difficulty_score += 1
            elif question_number > 100:
                difficulty_score += 0.5

        if difficulty_score >= 6:
            return "고급"
        elif difficulty_score >= 3:
            return "중급"
        else:
            return "초급"

    def analyze_question_complexity(self, question: str, question_number: int = None) -> float:
        """질문 복잡도 분석"""
        try:
            complexity_score = 0.0
            
            # 길이 기반 복잡도
            length_factor = min(len(question) / 400, 0.4)
            complexity_score += length_factor
            
            # 전문 용어 밀도
            technical_terms = re.findall(r'[A-Z]{2,}|법|조|항|규정|지침|위원회|기관', question)
            term_factor = min(len(technical_terms) / 8, 0.3)
            complexity_score += term_factor
            
            # 질문 유형별 복잡도
            if "설명하세요" in question or "기술하세요" in question:
                complexity_score += 0.15
            if "방안" in question or "절차" in question:
                complexity_score += 0.1
            
            # 위치별 복잡도 조정
            if question_number is not None:
                position_config = self._get_position_complexity_adjustment(question_number)
                complexity_score += position_config
            
            return min(complexity_score, 1.0)
        except Exception:
            return 0.5

    def _get_position_complexity_adjustment(self, question_number: int) -> float:
        """위치별 복잡도 조정"""
        if question_number <= 100:
            return 0.0
        elif question_number <= 300:
            return -0.05
        else:
            return 0.1

    def detect_english_response(self, text: str) -> bool:
        """영어 답변 감지"""
        if not text:
            return False
        
        try:
            english_words = re.findall(r'\b[a-zA-Z]+\b', text)
            if len(english_words) > 15:
                return True
            
            english_sentences = re.findall(r'[A-Z][a-zA-Z\s,\.]{30,}', text)
            if len(english_sentences) > 1:
                return True
                
            total_chars = len(re.sub(r'[^\w가-힣]', '', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            if total_chars > 0:
                english_ratio = english_chars / total_chars
                if english_ratio > 0.3:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"영어 답변 감지 오류: {e}")
            return False

    def detect_repetitive_patterns(self, text: str) -> bool:
        """반복 패턴 감지"""
        if not text or len(text) < 20:
            return False

        patterns = [
            r"(.{2,5})\s*(\1\s*){4,}",
            r"([가-힣]{1,2})\s*(\1\s*){4,}",
            r"(\w+\s+)(\1){3,}"
        ]

        for pattern in patterns:
            try:
                if re.search(pattern, text):
                    return True
            except Exception:
                continue

        words = text.split()
        if len(words) >= 6:
            word_counts = {}
            for word in words:
                if len(word) <= 2:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            for word, count in word_counts.items():
                if count >= 5:
                    return True

        return False

    def restore_korean_characters(self, text: str) -> str:
        """한국어 문자 복구"""
        if not text:
            return ""

        try:
            text = unicodedata.normalize("NFC", text)
        except Exception:
            pass

        for broken, correct in self.korean_recovery_mapping.items():
            text = text.replace(broken, correct)

        try:
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"[.,!?]{2,}", ".", text)
        except Exception:
            pass

        return text

    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0

        try:
            korean_chars = len(re.findall(r"[가-힣]", text))
            total_chars = len(re.sub(r"[^\w가-힣]", "", text))
            return korean_chars / total_chars if total_chars > 0 else 0.0
        except Exception:
            return 0.0

    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, 
                             question: str = "", question_number: int = None) -> bool:
        """한국어 답변 검증"""
        if not answer:
            return False

        answer = str(answer).strip()

        if self.detect_repetitive_patterns(answer):
            return False
            
        if self.detect_english_response(answer):
            return False

        if question_type == "multiple_choice":
            try:
                answer_num = int(answer)
                return 1 <= answer_num <= max_choice
            except ValueError:
                return False
        else:
            # 위치별 검증 기준 조정
            min_length = 12
            if question_number is not None and question_number > 300:
                min_length = 15
                
            if len(answer) < min_length:
                return False

            korean_ratio = self.calculate_korean_ratio(answer)
            required_ratio = 0.35
            if question_number is not None and question_number > 300:
                required_ratio = 0.4
                
            if korean_ratio < required_ratio:
                return False

            korean_chars = len(re.findall(r"[가-힣]", answer))
            if korean_chars < 8:
                return False

            meaningful_keywords = [
                "법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", "정책", 
                "체계", "시스템", "통제", "특징", "지표", "탐지", "대응", "기관",
                "위원회", "업무", "권한", "의무", "원칙",
            ]
            
            keyword_count = sum(1 for word in meaningful_keywords if word in answer)
            return keyword_count >= 2

    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, 
                              question_number: int = None) -> str:
        """한국어 답변 정규화"""
        if not answer:
            return ""

        answer = str(answer).strip()
        
        if self.detect_english_response(answer):
            return ""

        if question_type == "multiple_choice":
            number_patterns = [
                r'정답[:：]?\s*(\d+)',
                r'답[:：]?\s*(\d+)',
                r'\b(\d+)\b'
            ]
            
            for pattern in number_patterns:
                matches = re.findall(pattern, answer)
                for match in matches:
                    num = int(match)
                    if 1 <= num <= max_choice:
                        return str(num)
            return ""
        else:
            answer = self.restore_korean_characters(answer)
                
            if len(answer) < 10:
                return ""

            # 위치별 길이 제한 조정
            max_length = 600
            if question_number is not None and question_number > 300:
                max_length = 650

            if len(answer) > max_length:
                sentences = re.split(r'[.!?]', answer)
                truncated_sentences = []
                current_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and current_length + len(sentence) + 2 <= max_length:
                        truncated_sentences.append(sentence)
                        current_length += len(sentence) + 2
                    else:
                        break
                
                if truncated_sentences:
                    answer = ". ".join(truncated_sentences)
                    if not answer.endswith('.'):
                        answer += "."
                else:
                    answer = answer[:max_length-3] + "..."

            if answer and not answer.endswith((".", "다", "요", "함", "니다", "습니다")):
                if answer.endswith("니"):
                    answer += "다."
                elif answer.endswith("습"):
                    answer += "니다."
                elif answer.endswith(("해야", "필요", "있음")):
                    answer += "."
                else:
                    answer += "."

            return answer

    def cleanup(self):
        """리소스 정리"""
        pass