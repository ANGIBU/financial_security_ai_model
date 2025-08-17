# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
- 한국어 전용 처리
- 질문 의도 분석
- 강화된 패턴 매칭
"""

import re
import pickle
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from config import (
    PKL_DIR, KOREAN_REQUIREMENTS, JSON_CONFIG_FILES, 
    TEXT_CLEANUP_CONFIG, KOREAN_TYPO_MAPPING, OPTIMIZATION_CONFIG,
    TEXT_SAFETY_CONFIG, check_text_safety
)

class SimpleDataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        self.pkl_dir = PKL_DIR
        self.pkl_dir.mkdir(exist_ok=True)
        
        self._load_json_configs()
        
        self.korean_requirements = KOREAN_REQUIREMENTS
        self.text_cleanup_config = TEXT_CLEANUP_CONFIG
        self.korean_typo_mapping = KOREAN_TYPO_MAPPING
        self.text_safety_config = TEXT_SAFETY_CONFIG
        
        self.processing_stats = self.processing_stats_structure.copy()
        
        self._load_processing_history()
    
    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            with open(JSON_CONFIG_FILES['processing_config'], 'r', encoding='utf-8') as f:
                processing_config = json.load(f)
            
            self.mc_patterns = processing_config['mc_patterns']
            self.mc_keywords = processing_config['mc_keywords']
            self.question_intent_patterns = processing_config['question_intent_patterns']
            self.subj_patterns = processing_config['subj_patterns']
            self.processing_stats_structure = processing_config['processing_stats_structure']
            self.domain_detection_patterns = processing_config.get('domain_detection_patterns', {})
            self.choice_extraction_patterns = processing_config.get('choice_extraction_patterns', [])
            self.answer_validation_rules = processing_config.get('answer_validation_rules', {})
            self.high_frequency_keywords = processing_config.get('high_frequency_keywords', {})
            
            with open(JSON_CONFIG_FILES['knowledge_data'], 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            self.domain_keywords = knowledge_data['domain_keywords']
            
            print("데이터 처리 설정 파일 로드 완료")
            
        except FileNotFoundError as e:
            print(f"설정 파일을 찾을 수 없습니다: {e}")
            self._load_default_configs()
        except json.JSONDecodeError as e:
            print(f"JSON 파일 파싱 오류: {e}")
            self._load_default_configs()
        except Exception as e:
            print(f"설정 파일 로드 중 오류: {e}")
            self._load_default_configs()
    
    def _load_default_configs(self):
        """기본 설정 로드"""
        print("기본 설정으로 대체합니다.")
        
        self.mc_patterns = [
            r'1\s+[가-힣\w].*\n2\s+[가-힣\w].*\n3\s+[가-힣\w]',
            r'①.*②.*③.*④.*⑤'
        ]
        
        self.mc_keywords = [
            r'해당하지.*않는.*것',
            r'적절하지.*않는.*것',
            r'옳지.*않는.*것',
            r'맞는.*것',
            r'옳은.*것',
            r'적절한.*것'
        ]
        
        self.question_intent_patterns = {
            "기관_묻기": ["기관.*기술하세요", "기관.*설명하세요"],
            "특징_묻기": ["특징.*설명하세요", "특징.*기술하세요"],
            "지표_묻기": ["지표.*설명하세요", "탐지.*지표"],
            "방안_묻기": ["방안.*기술하세요", "방안.*설명하세요"],
            "절차_묻기": ["절차.*설명하세요", "절차.*기술하세요"],
            "조치_묻기": ["조치.*설명하세요", "조치.*기술하세요"]
        }
        
        self.subj_patterns = [
            r'설명하세요',
            r'기술하세요',
            r'서술하세요',
            r'작성하세요'
        ]
        
        self.domain_keywords = {
            "사이버보안": ["트로이", "RAT", "원격제어", "악성코드", "탐지", "지표"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원", "전자금융거래법"],
            "일반": ["법령", "규정", "관리", "조치", "절차"]
        }
        
        self.domain_detection_patterns = {}
        self.choice_extraction_patterns = []
        self.answer_validation_rules = {}
        self.high_frequency_keywords = {}
        
        self.processing_stats_structure = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "domain_distribution": {},
            "question_type_accuracy": {"correct": 0, "total": 0},
            "choice_count_errors": 0,
            "intent_analysis_accuracy": {"correct": 0, "total": 0},
            "intent_match_accuracy": {"correct": 0, "total": 0},
            "mc_classification_accuracy": {"correct": 0, "total": 0}
        }
    
    def _load_processing_history(self):
        """이전 처리 기록 로드"""
        history_file = self.pkl_dir / "processing_history.pkl"
        
        if history_file.exists():
            try:
                with open(history_file, 'rb') as f:
                    saved_stats = pickle.load(f)
                    self.processing_stats.update(saved_stats)
            except Exception:
                pass
    
    def _save_processing_history(self):
        """처리 기록 저장"""
        history_file = self.pkl_dir / "processing_history.pkl"
        
        try:
            save_data = {
                **self.processing_stats,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(history_file, 'wb') as f:
                pickle.dump(save_data, f)
        except Exception:
            pass
    
    def detect_text_corruption(self, text: str) -> bool:
        """텍스트 깨짐 감지 - 강화된 버전"""
        if not text:
            return True
        
        # config.py의 안전성 검사 활용
        return not check_text_safety(text)
    
    def clean_korean_text_premium(self, text: str) -> str:
        """프리미엄 한국어 텍스트 정리 - 안전성 우선"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 1단계: 깨진 텍스트 감지
        if self.detect_text_corruption(text):
            return ""  # 깨진 텍스트는 빈 문자열 반환
        
        # 2단계: 최소한의 안전한 정리만 수행
        # 안전한 한국어 오타 수정
        safe_corrections = {
            "전자금윋": "전자금융",
            "캉터": "컴퓨터",
            "하웨어": "하드웨어",
            "네됴크": "네트워크",
            "메세지": "메시지"
        }
        
        for wrong, correct in safe_corrections.items():
            text = text.replace(wrong, correct)
        
        # 3단계: 기본 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 4단계: 제어 문자 제거
        text = re.sub(r'[\u0000-\u001F\u007F]', '', text)
        
        return text
    
    def clean_korean_text_advanced(self, text: str) -> str:
        """한국어 텍스트 정리 - 안전 버전 사용"""
        return self.clean_korean_text_premium(text)
    
    def fix_korean_sentence_structure(self, text: str) -> str:
        """한국어 문장 구조 수정 - 안전 우선 버전"""
        if not text:
            return ""
        
        # 깨진 텍스트 확인
        if self.detect_text_corruption(text):
            return ""
        
        # 먼저 안전한 정리
        text = self.clean_korean_text_premium(text)
        if not text:
            return ""
        
        # 최소한의 안전한 문장 구조 수정
        text = text.strip()
        
        # 문장 끝 처리 (안전하게)
        if text and not text.endswith(('.', '다', '요', '함', '니다')):
            if len(text) > 15:
                text += '.'
        
        return text
    
    def analyze_question_intent(self, question: str) -> Dict:
        """강화된 질문 의도 분석 - 안정성 우선"""
        question_lower = question.lower()
        
        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],
            "context_hints": [],
            "domain_specific_intent": None
        }
        
        intent_scores = {}
        
        # 기관 관련 질문 우선 처리 (간소화된 패턴)
        institution_patterns = [
            r"기관.*기술하세요", r"기관.*설명하세요",
            r"어떤.*기관", r"어느.*기관", r"기관.*무엇",
            r"분쟁.*조정.*기관", r"신청.*수.*있는.*기관",
            r"담당.*기관", r"관리.*기관", r"감독.*기관"
        ]
        
        institution_score = 0
        for pattern in institution_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                institution_score += 2
        
        # 기관 키워드 추가 점수
        institution_keywords = [
            "전자금융", "분쟁조정", "개인정보", "침해신고", 
            "금융감독원", "한국은행", "개인정보보호위원회"
        ]
        
        for keyword in institution_keywords:
            if keyword in question_lower:
                institution_score += 1
        
        if institution_score >= 2:
            intent_scores["기관_묻기"] = {
                "score": institution_score,
                "patterns": ["institution_detection"]
            }
        
        # RAT/트로이 관련 질문 처리 (간소화)
        rat_patterns = [
            r"rat.*특징", r"트로이.*특징", r"원격.*제어.*특징",
            r"rat.*지표", r"트로이.*지표", r"원격.*제어.*지표",
            r"악성코드.*특징", r"악성코드.*지표"
        ]
        
        rat_score = 0
        for pattern in rat_patterns:
            if re.search(pattern, question_lower):
                rat_score += 2
        
        if rat_score > 0:
            if any(word in question_lower for word in ["특징", "성격"]):
                intent_scores["특징_묻기"] = {"score": rat_score + 1, "patterns": ["rat_feature"]}
            elif any(word in question_lower for word in ["지표", "징후", "탐지"]):
                intent_scores["지표_묻기"] = {"score": rat_score + 1, "patterns": ["rat_indicator"]}
        
        # 다른 의도 패턴 분석 (간소화)
        for intent_type, patterns in self.question_intent_patterns.items():
            if intent_type in intent_scores:
                continue  # 이미 처리됨
                
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns
                }
        
        # 최고 점수 의도 선택
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]
            
            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 3, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]
            
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명 필요")
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("특징과 성질 나열")
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 징후")
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("구체적 실행방안")
        
        self.processing_stats["intent_analysis_accuracy"]["total"] += 1
        
        return intent_analysis
    
    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """강화된 선택지 범위 추출"""
        question_type = self.analyze_question_type(question)
        
        if question_type != "multiple_choice":
            return "subjective", 0
        
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\s+(.+)', line)
            if match:
                num = int(match.group(1))
                content = match.group(2).strip()
                if 1 <= num <= 5 and len(content) > 0:
                    choice_numbers.append(num)
        
        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)
            
            expected_count = max_choice - min_choice + 1
            if (len(choice_numbers) == expected_count and 
                min_choice == 1 and 
                max_choice >= 3):
                return "multiple_choice", max_choice
        
        # 폴백 - 키워드 기반 판단
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                self.processing_stats["choice_count_errors"] += 1
                return "multiple_choice", 5
        
        return "subjective", 0
    
    def analyze_question_type(self, question: str) -> str:
        """강화된 질문 유형 분석"""
        
        question = question.strip()
        self.processing_stats["question_type_accuracy"]["total"] += 1
        self.processing_stats["mc_classification_accuracy"]["total"] += 1
        
        # 주관식 패턴 우선 확인 (강화)
        subjective_indicators = [
            r'설명하세요', r'기술하세요', r'서술하세요', r'작성하세요',
            r'제시하세요', r'논하시오', r'답하시오',
            r'특징.*설명', r'지표.*설명', r'방안.*기술',
            r'기관.*기술', r'절차.*설명', r'조치.*설명'
        ]
        
        for pattern in subjective_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"
        
        # 객관식 패턴 확인
        choice_pattern = r'\n(\d+)\s+[가-힣\w]'
        choice_matches = re.findall(choice_pattern, question)
        
        if len(choice_matches) >= 3:
            choice_nums = [int(match) for match in choice_matches]
            choice_nums.sort()
            if (choice_nums[0] == 1 and 
                len(choice_nums) == choice_nums[-1] and
                choice_nums[-1] <= 5):
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        # 키워드 기반 객관식 판단
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                if any(f'{i} ' in question for i in range(1, 6)):
                    self.processing_stats["question_type_accuracy"]["correct"] += 1
                    self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                    return "multiple_choice"
        
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """강화된 도메인 추출 - 안정성 우선"""
        question_lower = question.lower()
        
        domain_scores = {}
        
        # 정확한 도메인 매칭 (간소화된 패턴)
        domain_patterns = {
            "사이버보안": ["rat", "트로이", "원격제어", "원격접근", "악성코드", "탐지", "지표", "특징", "딥페이크", "sbom"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원", "전자금융거래법", "분쟁", "조정", "한국은행"],
            "개인정보보호": ["개인정보", "정보주체", "만 14세", "법정대리인", "개인정보보호법", "개인정보침해신고센터"],
            "정보보안": ["정보보안", "isms", "관리체계", "정책 수립", "재해복구", "bcp", "drp"],
            "금융투자": ["금융투자업", "투자자문", "투자매매", "금융투자", "자본시장법"],
            "위험관리": ["위험관리", "위험 관리", "재해복구", "위험수용", "위험평가"]
        }
        
        for domain, patterns in domain_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in question_lower:
                    # 핵심 키워드 가중치 적용
                    if pattern in ["rat", "트로이", "원격제어", "전자금융", "분쟁조정", "개인정보", "만 14세"]:
                        score += 3
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "일반"
        
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        return detected_domain
    
    def clean_korean_text(self, text: str) -> str:
        """한국어 전용 텍스트 정리 (안전 버전 사용)"""
        return self.clean_korean_text_premium(text)
    
    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def calculate_english_ratio(self, text: str) -> float:
        """영어 비율 계산"""
        if not text:
            return 0.0
        
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return english_chars / total_chars
    
    def validate_mc_answer_range(self, answer: str, max_choice: int) -> bool:
        """객관식 답변 범위 검증"""
        if not answer or not answer.isdigit():
            return False
        
        answer_num = int(answer)
        return 1 <= answer_num <= max_choice
    
    def validate_answer_intent_match(self, answer: str, question: str, intent_analysis: Dict) -> bool:
        """강화된 답변과 질문 의도 일치성 검증 - 안정성 우선"""
        if not answer or not intent_analysis:
            return False
        
        # 먼저 깨진 텍스트 검증
        if self.detect_text_corruption(answer):
            return False
        
        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()
        
        # 기관명 답변 검증 (간소화)
        if required_type == "기관명":
            institution_keywords = [
                "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원", 
                "조정위원회", "보호위원회", "신고센터", "분쟁조정"
            ]
            
            keyword_count = sum(1 for keyword in institution_keywords if keyword in answer_lower)
            role_keywords = ["담당", "업무", "수행", "관리", "감독", "조정", "신고", "접수"]
            role_count = sum(1 for keyword in role_keywords if keyword in answer_lower)
            
            match_found = keyword_count >= 1 or role_count >= 1
        
        elif required_type == "특징설명":
            feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할"]
            feature_count = sum(1 for keyword in feature_keywords if keyword in answer_lower)
            
            descriptive_words = ["위장", "은밀", "지속", "제어", "접근", "수행", "활동", "백도어"]
            desc_count = sum(1 for word in descriptive_words if word in answer_lower)
            
            match_found = feature_count >= 1 or (desc_count >= 2 and len(answer) > 30)
        
        elif required_type == "지표나열":
            indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "활동", "모니터링", "탐지"]
            indicator_count = sum(1 for keyword in indicator_keywords if keyword in answer_lower)
            
            specific_indicators = ["네트워크", "트래픽", "프로세스", "파일", "시스템", "로그"]
            specific_count = sum(1 for word in specific_indicators if word in answer_lower)
            
            match_found = indicator_count >= 1 or specific_count >= 2
        
        else:
            # 기본 유효성 - 의미있는 키워드 포함 여부
            meaningful_words = ["법령", "규정", "관리", "조치", "절차", "기준", "정책", "체계"]
            match_found = sum(1 for word in meaningful_words if word in answer_lower) >= 1
        
        self.processing_stats["intent_match_accuracy"]["total"] += 1
        if match_found:
            self.processing_stats["intent_match_accuracy"]["correct"] += 1
        
        return match_found
    
    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """강화된 한국어 답변 유효성 검증 - 안정성 우선"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            if not self.validate_mc_answer_range(answer, max_choice):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
        
        else:
            # 먼저 깨진 텍스트 검증
            if self.detect_text_corruption(answer):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 안전한 텍스트 정리
            clean_answer = self.clean_korean_text_premium(answer)
            
            if not clean_answer:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 관대한 검증 기준 적용
            min_length = 15  # 더 관대한 기준
            max_length = 400
            min_korean_ratio = 0.6  # 더 관대한 기준
            
            if not (min_length <= len(clean_answer) <= max_length):
                self.processing_stats["validation_failures"] += 1
                return False
            
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < min_korean_ratio:
                self.processing_stats["validation_failures"] += 1
                return False
            
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            if korean_chars < 10:  # 더 관대한 기준
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 의미있는 키워드 검증 (관대한 기준)
            meaningful_keywords = [
                "법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", 
                "정책", "체계", "시스템", "통제", "대응", "처리", "기관",
                "위원회", "특징", "지표", "탐지", "모니터링", "악성코드",
                "원격제어", "트로이", "분쟁조정", "개인정보", "전자금융"
            ]
            
            keyword_count = sum(1 for word in meaningful_keywords if word in clean_answer)
            if keyword_count < 1:
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
    
    def validate_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """답변 유효성 검증"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리 (안전 버전 사용)"""
        return self.clean_korean_text_premium(text)
    
    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화 - 안전 우선 버전"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            numbers = re.findall(r'[1-9]', answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            
            return ""
        
        else:
            # 안전한 텍스트 정리
            answer = self.clean_korean_text_premium(answer)
            
            # 정리 후 내용이 없으면 기본 답변
            if not answer:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            answer = self.fix_korean_sentence_structure(answer)
            
            # 너무 짧은 답변 처리
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 길이 제한 (안전하게)
            if len(answer) > 350:
                sentences = answer.split('. ')
                answer = '. '.join(sentences[:3])
                if len(answer) > 350:
                    answer = answer[:350]
            
            if answer and not answer.endswith(('.', '다', '요', '함')):
                answer += "."
            
            return answer
    
    def normalize_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """답변 정규화 (안전 버전 사용)"""
        return self.normalize_korean_answer(answer, question_type, max_choice)
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환"""
        total = max(self.processing_stats["total_processed"], 1)
        intent_total = max(self.processing_stats["intent_analysis_accuracy"]["total"], 1)
        intent_match_total = max(self.processing_stats["intent_match_accuracy"]["total"], 1)
        mc_total = max(self.processing_stats["mc_classification_accuracy"]["total"], 1)
        
        stats = {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "choice_count_errors": self.processing_stats["choice_count_errors"],
            "intent_analysis_accuracy_rate": (self.processing_stats["intent_analysis_accuracy"]["correct"] / intent_total) * 100,
            "intent_match_accuracy_rate": (self.processing_stats["intent_match_accuracy"]["correct"] / intent_match_total) * 100,
            "mc_classification_accuracy_rate": (self.processing_stats["mc_classification_accuracy"]["correct"] / mc_total) * 100,
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats["question_type_accuracy"]
        }
        
        return stats
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()