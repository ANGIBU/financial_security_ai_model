# data_processor.py

"""
데이터 처리기
- 객관식/주관식 분류
- 텍스트 정리
- 답변 검증
- 한국어 전용 처리
- 질문 의도 분석
- 선택지 의미 분석 강화
"""

import re
import pickle
import os
import json
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

# 설정 파일 import
from config import PKL_DIR, KOREAN_REQUIREMENTS, JSON_CONFIG_FILES

class SimpleDataProcessor:
    """데이터 처리기"""
    
    def __init__(self):
        # pkl 저장 폴더 생성
        self.pkl_dir = PKL_DIR
        self.pkl_dir.mkdir(exist_ok=True)
        
        # JSON 설정 파일에서 데이터 로드
        self._load_json_configs()
        
        # 한국어 전용 검증 기준 (config.py에서 로드)
        self.korean_requirements = KOREAN_REQUIREMENTS
        
        # 처리 통계
        self.processing_stats = self.processing_stats_structure.copy()
        
        # 이전 처리 기록 로드
        self._load_processing_history()
        
        # 선택지 의미 분석 강화를 위한 키워드 사전
        self._init_semantic_analysis_keywords()
    
    def _load_json_configs(self):
        """JSON 설정 파일들 로드"""
        try:
            # processing_config.json 로드
            with open(JSON_CONFIG_FILES['processing_config'], 'r', encoding='utf-8') as f:
                processing_config = json.load(f)
            
            # 데이터 처리 관련 설정 할당
            self.mc_patterns = processing_config['mc_patterns']
            self.mc_keywords = processing_config['mc_keywords']
            self.question_intent_patterns = processing_config['question_intent_patterns']
            self.subj_patterns = processing_config['subj_patterns']
            self.processing_stats_structure = processing_config['processing_stats_structure']
            
            # knowledge_data.json에서 도메인 키워드 로드
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
        """기본 설정 로드 (JSON 파일 로드 실패 시)"""
        print("기본 설정으로 대체합니다.")
        
        # 최소한의 기본 설정
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
            "일반": ["법령", "규정", "관리", "조치", "절차"]
        }
        
        self.processing_stats_structure = {
            "total_processed": 0,
            "korean_compliance": 0,
            "validation_failures": 0,
            "domain_distribution": {},
            "question_type_accuracy": {"correct": 0, "total": 0},
            "choice_count_errors": 0,
            "intent_analysis_accuracy": {"correct": 0, "total": 0},
            "intent_match_accuracy": {"correct": 0, "total": 0},
            "mc_classification_accuracy": {"correct": 0, "total": 0},
            "semantic_analysis_accuracy": {"correct": 0, "total": 0},
            "choice_content_extraction": {"success": 0, "total": 0},
            "negative_question_detection": {"correct": 0, "total": 0}
        }
    
    def _init_semantic_analysis_keywords(self):
        """의미 분석을 위한 키워드 사전 초기화"""
        self.semantic_keywords = {
            "금융투자": {
                "핵심업무": ["투자", "자문", "매매", "중개", "집합투자", "신탁"],
                "비핵심업무": ["보험", "소비자", "대출", "예금"],
                "업무구분": ["투자자문업", "투자매매업", "투자중개업", "집합투자업", "신탁업"],
                "비금융투자업": ["보험중개업", "소비자금융업", "은행업", "보험업"]
            },
            "위험관리": {
                "계획요소": ["대상", "기간", "범위", "목표", "전략"],
                "대응전략": ["회피", "수용", "전가", "감소", "완화"],
                "실행리소스": ["인력", "자원", "조직", "담당자", "예산"],
                "관리활동": ["모니터링", "평가", "보고", "검토", "개선"],
                "부적절요소": ["수행인력", "담당자", "예산배정", "조직구성"]
            },
            "개인정보보호": {
                "정책요소": ["경영진참여", "최고책임자", "자원할당", "조직구성"],
                "처리활동": ["수집", "이용", "제공", "파기", "보관"],
                "보호조치": ["암호화", "접근통제", "로그관리", "교육"],
                "권리보장": ["열람권", "정정삭제권", "처리정지권", "손해배상"]
            },
            "전자금융": {
                "요구사유": ["통화신용정책", "지급결제제도", "금융안정", "통계조사"],
                "기관역할": ["정책수립", "감독", "조정", "분쟁해결"],
                "거래보안": ["인증", "암호화", "접근매체", "전자서명"],
                "비관련사유": ["경영실적", "보안강화", "일반통계", "개인정보"]
            },
            "사이버보안": {
                "보안목적": ["투명성", "공급망보안", "취약점관리", "컴플라이언스"],
                "기술요소": ["소프트웨어구성", "라이선스", "의존성", "버전관리"],
                "비보안목적": ["접근제어", "개인정보보호", "다양성확보", "성능향상"]
            },
            "정보보안": {
                "복구요소": ["복구절차", "비상연락체계", "복구목표시간", "데이터백업"],
                "관리요소": ["정책수립", "위험평가", "보안조치", "교육훈련"],
                "비복구요소": ["개인정보파기", "일반업무절차", "성과평가", "예산계획"]
            }
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
    
    def analyze_question_intent_enhanced(self, question: str) -> Dict:
        """강화된 질문 의도 분석"""
        question_lower = question.lower()
        
        intent_analysis = {
            "primary_intent": "일반",
            "intent_confidence": 0.0,
            "detected_patterns": [],
            "answer_type_required": "설명형",
            "secondary_intents": [],
            "context_hints": [],
            "question_structure": {},
            "negative_indicators": [],
            "expected_answer_characteristics": {}
        }
        
        # 질문 구조 분석
        intent_analysis["question_structure"] = self._analyze_question_structure(question)
        
        # 부정형 질문 탐지 강화
        intent_analysis["negative_indicators"] = self._detect_negative_patterns(question)
        
        # 각 의도 패턴별 점수 계산
        intent_scores = {}
        
        for intent_type, patterns in self.question_intent_patterns.items():
            score = 0
            matched_patterns = []
            
            for pattern in patterns:
                matches = re.findall(pattern, question, re.IGNORECASE)
                if matches:
                    # 패턴 매칭 강도에 따른 점수 부여
                    if len(matches) > 1:
                        score += 2
                    else:
                        score += 1
                    matched_patterns.append(pattern)
            
            if score > 0:
                intent_scores[intent_type] = {
                    "score": score,
                    "patterns": matched_patterns
                }
        
        # 가장 높은 점수의 의도 선택
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            best_intent = sorted_intents[0]
            
            intent_analysis["primary_intent"] = best_intent[0]
            intent_analysis["intent_confidence"] = min(best_intent[1]["score"] / 3, 1.0)
            intent_analysis["detected_patterns"] = best_intent[1]["patterns"]
            
            # 부차적 의도들도 기록
            if len(sorted_intents) > 1:
                intent_analysis["secondary_intents"] = [
                    {"intent": intent, "score": data["score"]} 
                    for intent, data in sorted_intents[1:3]
                ]
            
            # 답변 유형 결정 강화
            primary = best_intent[0]
            if "기관" in primary:
                intent_analysis["answer_type_required"] = "기관명"
                intent_analysis["context_hints"].append("구체적인 기관명 필요")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["위원회", "기관", "감독원"],
                    "structure": "기관명 + 소속 + 역할",
                    "length_range": (50, 200)
                }
            elif "특징" in primary:
                intent_analysis["answer_type_required"] = "특징설명"
                intent_analysis["context_hints"].append("특징과 성질 나열")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["특징", "특성", "성질"],
                    "structure": "특징나열 + 설명",
                    "length_range": (80, 300)
                }
            elif "지표" in primary:
                intent_analysis["answer_type_required"] = "지표나열"
                intent_analysis["context_hints"].append("탐지 지표와 징후")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["지표", "징후", "탐지"],
                    "structure": "지표나열 + 활용방법",
                    "length_range": (100, 350)
                }
            elif "방안" in primary:
                intent_analysis["answer_type_required"] = "방안제시"
                intent_analysis["context_hints"].append("구체적 실행방안")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["방안", "대책", "조치"],
                    "structure": "방안제시 + 실행방법",
                    "length_range": (100, 400)
                }
            elif "절차" in primary:
                intent_analysis["answer_type_required"] = "절차설명"
                intent_analysis["context_hints"].append("단계별 절차")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["절차", "단계", "순서"],
                    "structure": "단계나열 + 설명",
                    "length_range": (80, 300)
                }
            elif "조치" in primary:
                intent_analysis["answer_type_required"] = "조치설명"
                intent_analysis["context_hints"].append("보안조치 내용")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["조치", "대응", "보안"],
                    "structure": "조치설명 + 효과",
                    "length_range": (70, 250)
                }
            elif "법령" in primary:
                intent_analysis["answer_type_required"] = "법령설명"
                intent_analysis["context_hints"].append("관련 법령과 규정")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["법", "법령", "규정"],
                    "structure": "법령명 + 내용",
                    "length_range": (60, 200)
                }
            elif "정의" in primary:
                intent_analysis["answer_type_required"] = "정의설명"
                intent_analysis["context_hints"].append("개념과 정의")
                intent_analysis["expected_answer_characteristics"] = {
                    "must_include": ["정의", "개념", "의미"],
                    "structure": "정의 + 설명",
                    "length_range": (50, 180)
                }
        
        # 추가 문맥 분석 강화
        self._add_enhanced_context_analysis(question, intent_analysis)
        
        # 통계 업데이트
        self.processing_stats["intent_analysis_accuracy"]["total"] += 1
        
        return intent_analysis
    
    def _analyze_question_structure(self, question: str) -> Dict:
        """질문 구조 분석"""
        structure = {
            "has_choices": False,
            "choice_count": 0,
            "question_type": "unknown",
            "negation_type": None,
            "focus_area": None,
            "complexity_level": "basic"
        }
        
        # 선택지 존재 여부 및 개수
        lines = question.split('\n')
        choice_count = 0
        for line in lines:
            match = re.match(r'^(\d+)\s+(.+)', line.strip())
            if match and 1 <= int(match.group(1)) <= 5:
                choice_count += 1
        
        structure["has_choices"] = choice_count >= 3
        structure["choice_count"] = choice_count
        structure["question_type"] = "multiple_choice" if structure["has_choices"] else "subjective"
        
        # 부정형 질문 유형 분석
        question_lower = question.lower()
        if "해당하지.*않는" in question_lower:
            structure["negation_type"] = "범주_예외"
        elif "적절하지.*않은" in question_lower:
            structure["negation_type"] = "부적절_요소"
        elif "옳지.*않는" in question_lower:
            structure["negation_type"] = "부정확_내용"
        
        # 질문 초점 영역
        if "구분" in question_lower and "해당하지" in question_lower:
            structure["focus_area"] = "카테고리_분류"
        elif "계획" in question_lower and "요소" in question_lower:
            structure["focus_area"] = "계획_구성"
        elif "가장.*중요한" in question_lower:
            structure["focus_area"] = "우선순위_판단"
        
        # 복잡도 수준
        technical_indicators = ["관리체계", "법령", "규정", "정책", "절차", "기준"]
        tech_count = sum(1 for indicator in technical_indicators if indicator in question_lower)
        
        if tech_count >= 3 or len(question) > 300:
            structure["complexity_level"] = "advanced"
        elif tech_count >= 1 or len(question) > 150:
            structure["complexity_level"] = "intermediate"
        
        return structure
    
    def _detect_negative_patterns(self, question: str) -> List[str]:
        """부정형 패턴 탐지"""
        indicators = []
        question_lower = question.lower()
        
        negative_patterns = [
            ("해당하지_않는", r"해당하지.*않는"),
            ("적절하지_않은", r"적절하지.*않은"),
            ("옳지_않은", r"옳지.*않은"),
            ("잘못된", r"잘못된"),
            ("틀린", r"틀린"),
            ("부적절한", r"부적절한"),
            ("예외", r"예외"),
            ("제외", r"제외")
        ]
        
        for pattern_name, pattern in negative_patterns:
            if re.search(pattern, question_lower):
                indicators.append(pattern_name)
        
        return indicators
    
    def _add_enhanced_context_analysis(self, question: str, intent_analysis: Dict):
        """강화된 추가 문맥 분석"""
        question_lower = question.lower()
        
        # 긴급성 표시어 확인
        urgency_keywords = ["긴급", "즉시", "신속", "빠른", "시급"]
        if any(keyword in question_lower for keyword in urgency_keywords):
            intent_analysis["context_hints"].append("긴급 대응 필요")
        
        # 예시 요구 확인
        example_keywords = ["예시", "사례", "구체적", "실제", "실무적"]
        if any(keyword in question_lower for keyword in example_keywords):
            intent_analysis["context_hints"].append("구체적 예시 포함")
        
        # 비교 요구 확인
        comparison_keywords = ["비교", "차이", "구별", "비교하여", "대비"]
        if any(keyword in question_lower for keyword in comparison_keywords):
            intent_analysis["context_hints"].append("비교 분석 필요")
        
        # 단계적 설명 요구 확인
        step_keywords = ["단계", "순서", "과정", "절차", "순차적"]
        if any(keyword in question_lower for keyword in step_keywords):
            intent_analysis["context_hints"].append("단계별 설명 필요")
        
        # 법적 근거 요구 확인
        legal_keywords = ["법령", "법률", "규정", "조항", "근거"]
        if any(keyword in question_lower for keyword in legal_keywords):
            intent_analysis["context_hints"].append("법적 근거 필요")
        
        # 도메인별 특화 분석
        domain = self.extract_domain(question)
        if domain in self.semantic_keywords:
            domain_analysis = self._analyze_domain_specific_context(question, domain)
            intent_analysis["context_hints"].extend(domain_analysis)
    
    def _analyze_domain_specific_context(self, question: str, domain: str) -> List[str]:
        """도메인별 특화 컨텍스트 분석"""
        hints = []
        question_lower = question.lower()
        
        if domain == "금융투자" and "구분" in question_lower:
            hints.append("금융투자업 카테고리 구분 필요")
            if "해당하지" in question_lower:
                hints.append("비금융투자업 식별 필요")
        
        elif domain == "위험관리" and "계획" in question_lower:
            hints.append("계획수립과 실행단계 구분 필요")
            if "적절하지" in question_lower:
                hints.append("계획외 요소 식별 필요")
        
        elif domain == "개인정보보호" and "정책" in question_lower:
            hints.append("정책수립 핵심요소 식별 필요")
            if "중요한" in question_lower:
                hints.append("우선순위 판단 필요")
        
        elif domain == "전자금융" and "요구" in question_lower:
            hints.append("법적 요구사유 확인 필요")
            if "경우" in question_lower:
                hints.append("적용조건 분석 필요")
        
        return hints
    
    def analyze_question_intent(self, question: str) -> Dict:
        """질문 의도 분석 (강화된 버전 사용)"""
        return self.analyze_question_intent_enhanced(question)
    
    def extract_choice_range(self, question: str) -> Tuple[str, int]:
        """선택지 범위 추출"""
        question_type = self.analyze_question_type_enhanced(question)
        
        if question_type != "multiple_choice":
            return "subjective", 0
        
        # 강화된 선택지 추출
        choices_info = self.extract_choices_with_semantic_analysis(question)
        
        if choices_info["valid_choice_count"] >= 3:
            return "multiple_choice", choices_info["max_choice_number"]
        
        # 전통적인 패턴으로 확인
        for i in range(5, 2, -1):
            # "1 텍스트 2 텍스트 3 텍스트" 패턴
            pattern_parts = [f'{j}\\s+[가-힣\\w]+' for j in range(1, i+1)]
            pattern = '.*'.join(pattern_parts)
            if re.search(pattern, question, re.DOTALL):
                return "multiple_choice", i
        
        # 객관식 키워드가 있지만 선택지를 찾을 수 없는 경우
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                self.processing_stats["choice_count_errors"] += 1
                return "multiple_choice", 5
        
        return "subjective", 0
    
    def extract_choices_with_semantic_analysis(self, question: str) -> Dict:
        """의미 분석을 포함한 선택지 추출"""
        choice_info = {
            "choices": {},
            "valid_choice_count": 0,
            "max_choice_number": 0,
            "semantic_categories": {},
            "outlier_candidates": [],
            "content_analysis": {}
        }
        
        lines = question.split('\n')
        
        # 선택지 추출
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\s+(.+)', line)
            if match:
                num = int(match.group(1))
                content = match.group(2).strip()
                if 1 <= num <= 5 and len(content) > 0:
                    choice_info["choices"][str(num)] = content
                    choice_info["valid_choice_count"] += 1
                    choice_info["max_choice_number"] = max(choice_info["max_choice_number"], num)
        
        # 의미 분석
        if choice_info["valid_choice_count"] >= 3:
            domain = self.extract_domain(question)
            choice_info["semantic_categories"] = self._categorize_choices(choice_info["choices"], domain)
            choice_info["outlier_candidates"] = self._find_outlier_choices(choice_info["choices"], domain)
            choice_info["content_analysis"] = self._analyze_choice_content_patterns(choice_info["choices"], domain)
        
        # 추출 성공률 업데이트
        self.processing_stats["choice_content_extraction"]["total"] += 1
        if choice_info["valid_choice_count"] >= 3:
            self.processing_stats["choice_content_extraction"]["success"] += 1
        
        return choice_info
    
    def _categorize_choices(self, choices: Dict[str, str], domain: str) -> Dict:
        """선택지 카테고리 분류"""
        categories = {}
        
        if domain not in self.semantic_keywords:
            return categories
        
        domain_keywords = self.semantic_keywords[domain]
        
        for choice_num, content in choices.items():
            content_lower = content.lower()
            
            # 각 카테고리별 점수 계산
            category_scores = {}
            for category, keywords in domain_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content_lower)
                if score > 0:
                    category_scores[category] = score
            
            # 가장 높은 점수의 카테고리 할당
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])[0]
                categories[choice_num] = best_category
            else:
                categories[choice_num] = "기타"
        
        return categories
    
    def _find_outlier_choices(self, choices: Dict[str, str], domain: str) -> List[str]:
        """이상치 선택지 찾기"""
        outliers = []
        
        if domain not in self.semantic_keywords:
            return outliers
        
        # 도메인 관련성 점수 계산
        relevance_scores = {}
        domain_keywords = self.semantic_keywords[domain]
        all_keywords = []
        for keywords in domain_keywords.values():
            all_keywords.extend(keywords)
        
        for choice_num, content in choices.items():
            content_lower = content.lower()
            relevance_score = sum(1 for keyword in all_keywords if keyword in content_lower)
            relevance_scores[choice_num] = relevance_score
        
        if relevance_scores:
            scores = list(relevance_scores.values())
            avg_score = sum(scores) / len(scores)
            
            # 평균의 절반 이하인 선택지를 이상치로 분류
            for choice_num, score in relevance_scores.items():
                if score < avg_score * 0.5:
                    outliers.append(choice_num)
        
        return outliers
    
    def _analyze_choice_content_patterns(self, choices: Dict[str, str], domain: str) -> Dict:
        """선택지 내용 패턴 분석"""
        analysis = {
            "length_distribution": {},
            "keyword_density": {},
            "structure_patterns": {},
            "domain_specificity": {}
        }
        
        for choice_num, content in choices.items():
            # 길이 분석
            analysis["length_distribution"][choice_num] = len(content)
            
            # 키워드 밀도
            if domain in self.semantic_keywords:
                all_keywords = []
                for keywords in self.semantic_keywords[domain].values():
                    all_keywords.extend(keywords)
                
                keyword_count = sum(1 for keyword in all_keywords if keyword in content.lower())
                analysis["keyword_density"][choice_num] = keyword_count / len(content.split()) if content.split() else 0
            
            # 구조 패턴
            if "업" in content:
                analysis["structure_patterns"][choice_num] = "업무_유형"
            elif any(word in content for word in ["요소", "사항", "내용"]):
                analysis["structure_patterns"][choice_num] = "구성_요소"
            elif any(word in content for word in ["방법", "방식", "절차"]):
                analysis["structure_patterns"][choice_num] = "처리_방법"
            else:
                analysis["structure_patterns"][choice_num] = "일반_내용"
        
        return analysis
    
    def analyze_question_type_enhanced(self, question: str) -> str:
        """강화된 질문 유형 분석"""
        
        question = question.strip()
        self.processing_stats["question_type_accuracy"]["total"] += 1
        self.processing_stats["mc_classification_accuracy"]["total"] += 1
        
        # 주관식 패턴 우선 확인
        for pattern in self.subj_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "subjective"
        
        # 강화된 객관식 확인
        choices_info = self.extract_choices_with_semantic_analysis(question)
        
        if choices_info["valid_choice_count"] >= 3:
            # 연속성 검증
            choice_nums = [int(num) for num in choices_info["choices"].keys()]
            choice_nums.sort()
            
            if (choice_nums[0] == 1 and 
                len(choice_nums) == choice_nums[-1] and
                choice_nums[-1] <= 5):
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        # 객관식 키워드 확인
        for pattern in self.mc_keywords:
            if re.search(pattern, question, re.IGNORECASE):
                # 선택지가 있는지 추가 확인
                if any(f'{i} ' in question for i in range(1, 6)):
                    self.processing_stats["question_type_accuracy"]["correct"] += 1
                    self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                    return "multiple_choice"
        
        # 전통적인 객관식 패턴 확인
        for pattern in self.mc_patterns:
            if re.search(pattern, question, re.DOTALL | re.MULTILINE):
                self.processing_stats["question_type_accuracy"]["correct"] += 1
                self.processing_stats["mc_classification_accuracy"]["correct"] += 1
                return "multiple_choice"
        
        # 길이와 구조 기반 추정
        if (len(question) < 400 and 
            re.search(r'것은\?|것\?|것은\s*$', question) and
            len(re.findall(r'\b[1-5]\b', question)) >= 3):
            return "multiple_choice"
        
        return "subjective"
    
    def analyze_question_type(self, question: str) -> str:
        """질문 유형 분석 (강화된 버전 사용)"""
        return self.analyze_question_type_enhanced(question)
    
    def extract_domain(self, question: str) -> str:
        """도메인 추출"""
        question_lower = question.lower()
        
        # 각 도메인별 키워드 매칭 점수 계산
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in question_lower:
                    # 핵심 키워드는 가중치 부여
                    if keyword in ["개인정보보호법", "전자금융거래법", "자본시장법", "ISMS", 
                                  "트로이", "RAT", "원격제어", "분쟁조정", "위험관리"]:
                        score += 3
                    else:
                        score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        if not domain_scores:
            return "일반"
        
        # 가장 높은 점수의 도메인 반환
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # 실제 데이터 분포에 맞는 추가 검증
        if detected_domain == "사이버보안":
            cybersec_keywords = ["트로이", "악성코드", "RAT", "원격제어", "딥페이크", "SBOM", "보안"]
            if any(keyword in question_lower for keyword in cybersec_keywords):
                detected_domain = "사이버보안"
        elif detected_domain == "개인정보보호":
            privacy_keywords = ["개인정보", "정보주체", "만 14세", "법정대리인", "PIMS"]
            if any(keyword in question_lower for keyword in privacy_keywords):
                detected_domain = "개인정보보호"
        
        # 통계 업데이트
        if detected_domain not in self.processing_stats["domain_distribution"]:
            self.processing_stats["domain_distribution"][detected_domain] = 0
        self.processing_stats["domain_distribution"][detected_domain] += 1
        
        return detected_domain
    
    def clean_korean_text(self, text: str) -> str:
        """한국어 전용 텍스트 정리"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 깨진 문자 및 인코딩 오류 처리
        text = re.sub(r'[^\w\s가-힣.,!?()[\]\-]', ' ', text)
        
        # 영어 문자 제거
        text = re.sub(r'[a-zA-Z]+', '', text)
        
        # 중국어 제거
        text = re.sub(r'[\u4e00-\u9fff]', '', text)
        
        # 특수 기호 제거
        text = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', text)
        
        # 반복 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
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
    
    def validate_answer_intent_match_enhanced(self, answer: str, question: str, intent_analysis: Dict) -> bool:
        """강화된 답변과 질문 의도 일치성 검증"""
        if not answer or not intent_analysis:
            return False
        
        required_type = intent_analysis.get("answer_type_required", "설명형")
        answer_lower = answer.lower()
        expected_chars = intent_analysis.get("expected_answer_characteristics", {})
        
        # 기본 길이 검증
        if "length_range" in expected_chars:
            min_len, max_len = expected_chars["length_range"]
            if not (min_len <= len(answer) <= max_len):
                return False
        
        # 필수 포함 키워드 검증
        if "must_include" in expected_chars:
            must_keywords = expected_chars["must_include"]
            found_count = sum(1 for keyword in must_keywords if keyword in answer_lower)
            if found_count == 0:
                return False
        
        # 유형별 세부 검증
        match_found = self._validate_answer_type_match(answer_lower, required_type)
        
        # 통계 업데이트
        self.processing_stats["intent_match_accuracy"]["total"] += 1
        if match_found:
            self.processing_stats["intent_match_accuracy"]["correct"] += 1
        
        return match_found
    
    def _validate_answer_type_match(self, answer_lower: str, answer_type: str) -> bool:
        """답변 유형별 매칭 검증"""
        if answer_type == "기관명":
            # 구체적인 기관명이 포함되어야 하고, 최소 2개의 키워드 필요
            institution_keywords = [
                "위원회", "감독원", "은행", "기관", "센터", "청", "부", "원", 
                "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
                "한국은행", "금융위원회", "과학기술정보통신부", "개인정보침해신고센터"
            ]
            
            # 구체적 기관명 확인
            specific_institutions = [
                "전자금융분쟁조정위원회", "금융감독원", "개인정보보호위원회",
                "개인정보침해신고센터", "한국은행", "금융위원회"
            ]
            
            has_specific = any(inst in answer_lower for inst in specific_institutions)
            keyword_count = sum(1 for keyword in institution_keywords if keyword in answer_lower)
            
            return has_specific and keyword_count >= 2
        
        elif answer_type == "특징설명":
            feature_keywords = ["특징", "특성", "속성", "성질", "기능", "역할", "원리", "성격"]
            descriptive_words = ["위장", "은밀", "지속", "제어", "접근", "수행", "활동"]
            
            feature_count = sum(1 for keyword in feature_keywords if keyword in answer_lower)
            desc_count = sum(1 for word in descriptive_words if word in answer_lower)
            
            return feature_count >= 1 and desc_count >= 2
        
        elif answer_type == "지표나열":
            indicator_keywords = ["지표", "신호", "징후", "패턴", "행동", "활동", "모니터링", "탐지", "발견", "식별"]
            specific_indicators = ["네트워크", "트래픽", "프로세스", "파일", "시스템", "로그", "연결"]
            
            indicator_count = sum(1 for keyword in indicator_keywords if keyword in answer_lower)
            specific_count = sum(1 for word in specific_indicators if word in answer_lower)
            
            return indicator_count >= 2 and specific_count >= 2
        
        elif answer_type == "방안제시":
            solution_keywords = ["방안", "대책", "조치", "해결", "대응", "관리", "처리", "절차", "개선", "예방"]
            action_words = ["수립", "구축", "시행", "실시", "강화", "개선", "마련"]
            
            solution_count = sum(1 for keyword in solution_keywords if keyword in answer_lower)
            action_count = sum(1 for word in action_words if word in answer_lower)
            
            return solution_count >= 2 and action_count >= 1
        
        elif answer_type == "절차설명":
            procedure_keywords = ["절차", "과정", "단계", "순서", "프로세스", "진행", "수행", "실행"]
            step_indicators = ["첫째", "둘째", "먼저", "다음", "마지막", "단계적", "순차적"]
            
            proc_count = sum(1 for keyword in procedure_keywords if keyword in answer_lower)
            step_count = sum(1 for word in step_indicators if word in answer_lower)
            
            return proc_count >= 1 and (step_count >= 1 or "," in answer_lower)
        
        elif answer_type == "조치설명":
            measure_keywords = ["조치", "대응", "대책", "방안", "보안", "예방", "개선", "강화", "보완"]
            return sum(1 for keyword in measure_keywords if keyword in answer_lower) >= 2
        
        elif answer_type == "법령설명":
            law_keywords = ["법", "법령", "법률", "규정", "조항", "규칙", "기준", "근거"]
            return sum(1 for keyword in law_keywords if keyword in answer_lower) >= 2
        
        elif answer_type == "정의설명":
            definition_keywords = ["정의", "개념", "의미", "뜻", "용어", "개념"]
            return sum(1 for keyword in definition_keywords if keyword in answer_lower) >= 1
        
        else:
            # 기본적으로 통과
            meaningful_words = ["법령", "규정", "관리", "조치", "절차", "기준", "정책", "체계", "시스템"]
            return sum(1 for word in meaningful_words if word in answer_lower) >= 2
    
    def validate_answer_intent_match(self, answer: str, question: str, intent_analysis: Dict) -> bool:
        """답변과 질문 의도 일치성 검증 (강화된 버전 사용)"""
        return self.validate_answer_intent_match_enhanced(answer, question, intent_analysis)
    
    def validate_korean_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """한국어 답변 유효성 검증"""
        if not answer:
            return False
        
        answer = str(answer).strip()
        self.processing_stats["total_processed"] += 1
        
        if question_type == "multiple_choice":
            # 객관식: 지정된 범위의 숫자
            if not self.validate_mc_answer_range(answer, max_choice):
                self.processing_stats["validation_failures"] += 1
                return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
        
        else:
            # 주관식: 한국어 전용 검증
            clean_answer = self.clean_korean_text(answer)
            
            # 길이 검증
            if not (self.korean_requirements["min_length"] <= len(clean_answer) <= self.korean_requirements["max_length"]):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 한국어 비율 검증
            korean_ratio = self.calculate_korean_ratio(clean_answer)
            if korean_ratio < self.korean_requirements["min_korean_ratio"]:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 영어 비율 검증
            english_ratio = self.calculate_english_ratio(answer)
            if english_ratio > self.korean_requirements["max_english_ratio"]:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 최소 한국어 문자 수 검증
            korean_chars = len(re.findall(r'[가-힣]', clean_answer))
            if korean_chars < 20:
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 의미 있는 내용인지 확인
            meaningful_keywords = ["법", "규정", "조치", "관리", "보안", "방안", "절차", "기준", "정책", "체계", "시스템", "통제"]
            if not any(word in clean_answer for word in meaningful_keywords):
                self.processing_stats["validation_failures"] += 1
                return False
            
            # 질문 의도 일치성 검증 강화
            if question:
                intent_analysis = self.analyze_question_intent(question)
                if not self.validate_answer_intent_match(answer, question, intent_analysis):
                    self.processing_stats["validation_failures"] += 1
                    return False
            
            self.processing_stats["korean_compliance"] += 1
            return True
    
    def validate_answer(self, answer: str, question_type: str, max_choice: int = 5, question: str = "") -> bool:
        """답변 유효성 검증"""
        return self.validate_korean_answer(answer, question_type, max_choice, question)
    
    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        return self.clean_korean_text(text)
    
    def extract_choices(self, question: str) -> List[str]:
        """객관식 선택지 추출"""
        choices_info = self.extract_choices_with_semantic_analysis(question)
        
        if choices_info["valid_choice_count"] >= 3:
            # 번호 순서대로 정렬하여 반환
            choice_list = []
            for i in range(1, choices_info["max_choice_number"] + 1):
                if str(i) in choices_info["choices"]:
                    choice_list.append(choices_info["choices"][str(i)])
            return choice_list
        
        # 폴백: 전통적인 패턴으로도 확인
        choices = []
        if not choices:
            patterns = [
                r'(\d+)\s+([^0-9\n]+?)(?=\d+\s+|$)',
                r'(\d+)\)\s*([^0-9\n]+?)(?=\d+\)|$)',
                r'(\d+)\.\s*([^0-9\n]+?)(?=\d+\.|$)',
                r'[①②③④⑤]\s*([^①②③④⑤\n]+?)(?=[①②③④⑤]|$)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
                if matches:
                    if isinstance(matches[0], tuple):
                        choices = [match[1].strip() for match in matches]
                    else:
                        choices = [match.strip() for match in matches]
                    
                    if len(choices) >= 3:
                        break
        
        return choices[:5]
    
    def analyze_question_difficulty(self, question: str) -> str:
        """질문 난이도 분석"""
        question_lower = question.lower()
        
        # 전문 용어 개수
        technical_terms = [
            "isms", "pims", "sbom", "원격제어", "침입탐지", 
            "트로이", "멀웨어", "랜섬웨어", "딥페이크", "피싱",
            "접근매체", "전자서명", "개인정보보호법", "자본시장법",
            "rat", "원격접근", "탐지지표", "apt", "ddos",
            "ids", "ips", "bcp", "drp", "isms-p", "분쟁조정",
            "금융투자업", "위험관리", "재해복구", "비상연락체계"
        ]
        
        term_count = sum(1 for term in technical_terms if term in question_lower)
        
        # 문장 길이
        length = len(question)
        
        # 선택지 개수
        choice_count = len(self.extract_choices(question))
        
        # 난이도 계산
        if term_count >= 3 or length > 400 or choice_count >= 5:
            return "고급"
        elif term_count >= 1 or length > 200 or choice_count >= 4:
            return "중급"
        else:
            return "초급"
    
    def normalize_korean_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """한국어 답변 정규화"""
        if not answer:
            return ""
        
        answer = str(answer).strip()
        
        if question_type == "multiple_choice":
            # 숫자만 추출하고 범위 검증
            numbers = re.findall(r'[1-9]', answer)
            for num in numbers:
                if 1 <= int(num) <= max_choice:
                    return num
            
            return ""
        
        else:
            # 주관식 답변 한국어 정리
            answer = self.clean_korean_text(answer)
            
            # 의미 없는 짧은 문장 제거
            if len(answer) < 20:
                return "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다."
            
            # 길이 제한
            if len(answer) > self.korean_requirements["max_length"]:
                sentences = answer.split('. ')
                answer = '. '.join(sentences[:3])
                if len(answer) > self.korean_requirements["max_length"]:
                    answer = answer[:self.korean_requirements["max_length"]]
            
            # 마침표 확인
            if answer and not answer.endswith(('.', '다', '요', '함')):
                answer += "."
            
            return answer
    
    def normalize_answer(self, answer: str, question_type: str, max_choice: int = 5) -> str:
        """답변 정규화"""
        return self.normalize_korean_answer(answer, question_type, max_choice)
    
    def get_processing_stats(self) -> Dict:
        """처리 통계 반환"""
        total = max(self.processing_stats["total_processed"], 1)
        intent_total = max(self.processing_stats["intent_analysis_accuracy"]["total"], 1)
        intent_match_total = max(self.processing_stats["intent_match_accuracy"]["total"], 1)
        mc_total = max(self.processing_stats["mc_classification_accuracy"]["total"], 1)
        semantic_total = max(self.processing_stats.get("semantic_analysis_accuracy", {"total": 1})["total"], 1)
        choice_extract_total = max(self.processing_stats["choice_content_extraction"]["total"], 1)
        negative_total = max(self.processing_stats.get("negative_question_detection", {"total": 1})["total"], 1)
        
        return {
            "total_processed": self.processing_stats["total_processed"],
            "korean_compliance_rate": (self.processing_stats["korean_compliance"] / total) * 100,
            "validation_failure_rate": (self.processing_stats["validation_failures"] / total) * 100,
            "choice_count_errors": self.processing_stats["choice_count_errors"],
            "intent_analysis_accuracy_rate": (self.processing_stats["intent_analysis_accuracy"]["correct"] / intent_total) * 100,
            "intent_match_accuracy_rate": (self.processing_stats["intent_match_accuracy"]["correct"] / intent_match_total) * 100,
            "mc_classification_accuracy_rate": (self.processing_stats["mc_classification_accuracy"]["correct"] / mc_total) * 100,
            "semantic_analysis_accuracy_rate": (self.processing_stats.get("semantic_analysis_accuracy", {"correct": 0})["correct"] / semantic_total) * 100,
            "choice_content_extraction_rate": (self.processing_stats["choice_content_extraction"]["success"] / choice_extract_total) * 100,
            "negative_question_detection_rate": (self.processing_stats.get("negative_question_detection", {"correct": 0})["correct"] / negative_total) * 100,
            "domain_distribution": dict(self.processing_stats["domain_distribution"]),
            "question_type_accuracy": self.processing_stats["question_type_accuracy"]
        }
    
    def get_korean_requirements(self) -> Dict:
        """한국어 요구사항 반환"""
        return dict(self.korean_requirements)
    
    def cleanup(self):
        """정리"""
        self._save_processing_history()