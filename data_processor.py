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
    """데이터 처리기 - 안정성 강화 버전"""
    
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
            "지표_묻기": ["지표.*설명하세요", "탐지.*지표"]
        }
        
        self.subj_patterns = [
            r'설명하세요',
            r'기술하세요',
            r'서술하세요'
        ]
        
        self.domain_keywords = {
            "사이버보안": ["트로이", "RAT", "원격제어", "악성코드"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원"],
            "일반": ["법령", "규정", "관리", "조치"]
        }
        
        self.domain_detection_patterns = {}
        self.choice_extraction_patterns = []
        self.answer_validation_rules = {}
        
        self.processing_stats_structure = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "domain_distribution": {},
            "question_type_accuracy": {"correct": 0, "total": 0}
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
        """프리미엄 한국어 텍스트 정리 - 안전성 최우선"""
        if not text:
            return ""
        
        text = str(text).strip()
        
        # 1단계: 깨진 텍스트 감지
        if self.detect_text_corruption(text):
            return ""  # 깨진 텍스트는 빈 문자열 반환
        
        # 2단계: 안전한 한국어 오타 수정
        safe_corrections = {
            "전자금윋": "전자금융",
            "캉터": "컴퓨터",
            "하웨어": "하드웨어",
            "네됴크": "네트워크",
            "메세지": "메시지",
            "트래픁": "트래픽",
            "보안조최": "보안조치",
            "관리방안": "관리 방안"
        }
        
        for wrong, correct in safe_corrections.items():
            text = text.replace(wrong, correct)
        
        # 3단계: 기본 공백 정리
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 4단계: 제어 문자 제거
        text = re.sub(r'[\u0000-\u001F\u007F]', '', text)
        
        # 5단계: 문장 끝 정리
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
            "context_hints": []
        }
        
        intent_scores = {}
        
        # 기관 관련 질문 우선 처리 (간소화된 패턴)
        institution_patterns = [
            r"기관.*기술하세요", r"기관.*설명하세요",
            r"어떤.*기관", r"어느.*기관", r"기관.*무엇",
            r"분쟁.*조정.*기관", r"신청.*수.*있는.*기관",
            r"담당.*기관", r"관리.*기관"
        ]
        
        institution_score = 0
        for pattern in institution_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                institution_score += 2
        
        # 기관 키워드 추가 점수
        institution_keywords = [
            "전자금융", "분쟁조정", "개인정보", "침해신고", 
            "금융감독원", "개인정보보호위원회"
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
        
        # 기본 의도 패턴 분석 (간소화)
        basic_intent_patterns = {
            "특징_묻기": [r"특징.*설명", r"특징.*기술", r"어떤.*특징"],
            "지표_묻기": [r"지표.*설명", r"탐지.*지표", r"주요.*지표"],
            "방안_묻기": [r"방안.*기술", r"대응.*방안", r"해결.*방안"],
            "절차_묻기": [r"절차.*설명", r"과정.*설명", r"단계.*설명"],
            "조치_묻기": [r"조치.*설명", r"보안.*조치", r"대응.*조치"]
        }
        
        for intent_type, patterns in basic_intent_patterns.items():
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
            
            # 답변 유형 설정
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
        
        return intent_analysis
    
    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """강화된 선택지 범위 추출"""
        question_type = self.analyze_question_type(question)
        
        if question_type != "multiple_choice":
            return "subjective", 0
        
        # 개선된 선택지 추출
        lines = question.split('\n')
        choice_numbers = []
        
        for line in lines:
            line = line.strip()
            # 다양한 패턴 지원
            patterns = [
                r'^(\d+)\s+(.+)',  # 기본: 숫자 + 공백 + 내용
                r'^(\d+)\.\s*(.+)',  # 숫자 + 점 + 공백 + 내용
                r'^(\d+)\)\s*(.+)'   # 숫자 + 괄호 + 공백 + 내용
            ]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    num = int(match.group(1))
                    content = match.group(2).strip()
                    if 1 <= num <= 5 and len(content) > 0:
                        choice_numbers.append(num)
                    break
        
        if choice_numbers:
            choice_numbers.sort()
            max_choice = max(choice_numbers)
            min_choice = min(choice_numbers)
            
            # 연속성 검증
            expected_count = max_choice - min_choice + 1
            if (len(choice_numbers) == expected_count and 
                min_choice == 1 and 
                max_choice >= 3):
                return "multiple_choice", max_choice
        
        # 폴백 - 키워드 기반 판단
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                return "multiple_choice", 5  # 기본값
        
        return "subjective", 0
    
    def analyze_question_type(self, question: str) -> str:
        """강화된 질문 유형 분석"""
        
        question = question.strip()
        
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
        
        # 객관식 패턴 확인 (개선된 검증)
        choice_pattern = r'\n(\d+)\s+[가-힣\w]'
        choice_matches = re.findall(choice_pattern, question)
        
        if len(choice_matches) >= 3:
            choice_nums = [int(match) for match in choice_matches]
            choice_nums.sort()
            if (choice_nums[0] == 1 and 
                len(choice_nums) == choice_nums[-1] and
                choice_nums[-1] <= 5):
                return "multiple_choice"
        
        # 키워드 기반 객관식 판단
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 선택지가 실제로 있는지 확인
                if any(f'{i} ' in question for i in range(1, 6)):
                    return "multiple_choice"
        
        return "subjective"
    
    def extract_domain(self, question: str) -> str:
        """강화된 도메인 추출 - 안정성 우선"""
        question_lower = question.lower()
        
        domain_scores = {}
        
        # 정확한 도메인 매칭 (핵심 키워드 기반)
        domain_patterns = {
            "사이버보안": ["rat", "트로이", "원격제어", "원격접근", "악성코드", "딥페이크", "sbom"],
            "전자금융": ["전자금융", "분쟁조정", "금융감독원", "한국은행"],
            "개인정보보호": ["개인정보", "정보주체", "만 14세", "법정대리인", "개인정보보호위원회"],
            "정보보안": ["정보보안", "isms", "관리체계", "재해복구"],
            "금융투자": ["금융투자업", "투자자문", "투자매매", "자본시장법"],
            "위험관리": ["위험관리", "위험수용", "재해복구", "위험평가"]
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
        
        # 통계 업데이트
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        return detected_domain
    
    def calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """강화된 한국어 답변 유효성 검증 - 안정성 우선"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            # 객관식: 숫자 범위 검증
            if not (answer.isdigit() and 1 <= int(answer) <= max_choice):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
        
        else:
            # 주관식: 깨진 텍스트 우선 검증
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
            if korean_chars < 8:  # 더 관대한 기준
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
    
    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화 - 안전 우선 버전"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 객관식: 숫자 추출
            numbers = re.findall(r'[1-9]', answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            
            return ""
        
        else:
            # 주관식: 안전한 텍스트 정리
            answer = self.clean_korean_text_premium(answer)
            
            # 정리 후 내용이 없으면 기본 답변
            if not answer:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 너무 짧은 답변 처리
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 길이 제한 (안전하게)
            if len(answer) > 350:
                sentences = answer.split('. ')
                answer = '. '.join(sentences[:3])
                if len(answer) > 350:
                    answer = answer[:350]
            
            # 문장 끝 처리
            if answer and not answer.endswith(('.', '다', '요', '함')):
                answer += "."
            
            return answer
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환"""
        total = max(self.processing_stats["total_processed"], 1)
        
        stats = {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats.get("question_type_accuracy", {"correct": 0, "total": 0})
        }
        
        return stats
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()