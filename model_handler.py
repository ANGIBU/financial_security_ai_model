# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
- 학습 데이터 저장
"""

import torch
import re
import time
import gc
import random
import pickle
import os
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings
warnings.filterwarnings("ignore")

class SimpleModelHandler:
    """모델 핸들러"""
    
    def __init__(self, model_name: str = "upstage/SOLAR-10.7B-Instruct-v1.0", verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 답변 분포 추적
        self.answer_distribution = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        self.total_mc_answers = 0
        
        # 학습 데이터 저장
        self.learning_data = {
            "successful_answers": [],
            "failed_answers": [],
            "question_patterns": {},
            "answer_quality_scores": []
        }
        
        # 이전 학습 데이터 로드
        self._load_learning_data()
        
        print(f"모델 로딩: {model_name}")
        print(f"디바이스: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        
        # 한국어 전용 주관식 답변 템플릿
        self.korean_templates = {
            "개인정보보호": [
                "개인정보보호법에 따라 정보주체의 권리를 보장하고 개인정보처리자는 수집부터 파기까지 전 과정에서 적절한 보호조치를 이행해야 합니다.",
                "개인정보 처리 시 정보주체의 동의를 받고 목적 범위 내에서만 이용하며 개인정보보호위원회의 기준에 따른 안전성 확보조치를 수립해야 합니다.",
                "정보주체는 개인정보 열람권, 정정삭제권, 처리정지권을 가지며 개인정보처리자는 이러한 권리 행사를 보장하는 절차를 마련해야 합니다."
            ],
            "전자금융": [
                "전자금융거래법에 따라 전자금융업자는 이용자의 전자금융거래 안전성 확보를 위한 보안조치를 시행하고 금융감독원의 감독을 받아야 합니다.",
                "전자금융분쟁조정위원회에서 전자금융거래 분쟁조정 업무를 담당하며 이용자는 관련 법령에 따라 분쟁조정을 신청할 수 있습니다.",
                "전자금융서비스 제공 시 접근매체에 대한 보안성을 확보하고 이용자 인증 절차를 통해 거래의 안전성을 보장해야 합니다."
            ],
            "사이버보안": [
                "해당 악성코드는 원격제어 기능을 통해 시스템에 침입하며 백신 프로그램과 행위 기반 탐지 시스템을 활용하여 탐지할 수 있습니다.",
                "사이버보안 위협에 대응하기 위해서는 다층 방어체계를 구축하고 실시간 모니터링과 침입탐지시스템을 운영해야 합니다.",
                "보안정책을 수립하고 정기적인 보안교육과 훈련을 실시하며 취약점 점검과 보안패치를 지속적으로 수행해야 합니다."
            ],
            "정보보안": [
                "정보보안 관리체계 구축을 위해 보안정책 수립, 위험분석, 보안대책 구현, 사후관리의 절차를 체계적으로 운영해야 합니다.",
                "접근통제 정책을 수립하고 사용자별 권한을 관리하며 로그 모니터링과 정기적인 보안감사를 통해 보안수준을 유지해야 합니다.",
                "보안관제센터를 운영하고 침입탐지시스템과 방화벽을 통해 실시간 보안위협을 탐지하고 대응해야 합니다."
            ],
            "금융투자": [
                "자본시장법에 따라 금융투자업자는 투자자 보호와 시장 공정성 확보를 위한 내부통제기준을 수립하고 준수해야 합니다.",
                "금융투자업 영위 시 투자자의 투자성향과 위험도를 평가하고 적합한 상품을 권유하는 적합성 원칙을 준수해야 합니다.",
                "펀드 운용 시 투자자에게 투자위험과 손실 가능성을 충분히 설명하고 투명한 정보공시 의무를 이행해야 합니다."
            ],
            "위험관리": [
                "위험관리 체계 구축을 위해 위험식별, 위험평가, 위험대응, 위험모니터링의 단계별 절차를 수립하고 운영해야 합니다.",
                "내부통제시스템을 구축하고 정기적인 위험평가를 실시하여 잠재적 위험요소를 사전에 식별하고 대응방안을 마련해야 합니다.",
                "컴플라이언스 체계를 수립하고 법규 준수 현황을 점검하며 위반 시 즉시 시정조치를 취하는 관리체계를 운영해야 합니다."
            ]
        }
        
        # 워밍업
        self._warmup()
        
        print("모델 로딩 완료")
        
        # 학습 데이터 로드 현황
        if len(self.learning_data["successful_answers"]) > 0:
            print(f"이전 학습 데이터 로드: 성공 {len(self.learning_data['successful_answers'])}개, 실패 {len(self.learning_data['failed_answers'])}개")
    
    def _load_learning_data(self):
        """이전 학습 데이터 로드"""
        learning_file = "./learning_data.pkl"
        
        if os.path.exists(learning_file):
            try:
                with open(learning_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.learning_data.update(saved_data)
                if self.verbose:
                    print("학습 데이터 로드 완료")
            except Exception as e:
                if self.verbose:
                    print(f"학습 데이터 로드 오류: {e}")
    
    def _save_learning_data(self):
        """학습 데이터 저장"""
        learning_file = "./learning_data.pkl"
        
        try:
            # 저장할 데이터 정리 (최근 1000개까지만)
            save_data = {
                "successful_answers": self.learning_data["successful_answers"][-1000:],
                "failed_answers": self.learning_data["failed_answers"][-500:],
                "question_patterns": self.learning_data["question_patterns"],
                "answer_quality_scores": self.learning_data["answer_quality_scores"][-1000:],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(learning_file, 'wb') as f:
                pickle.dump(save_data, f)
                
            if self.verbose:
                print("학습 데이터 저장 완료")
        except Exception as e:
            if self.verbose:
                print(f"학습 데이터 저장 오류: {e}")
    
    def _add_learning_record(self, question: str, answer: str, question_type: str, success: bool, quality_score: float = 0.0):
        """학습 기록 추가"""
        record = {
            "question": question[:200],  # 질문 요약
            "answer": answer[:300],      # 답변 요약
            "type": question_type,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score
        }
        
        if success:
            self.learning_data["successful_answers"].append(record)
        else:
            self.learning_data["failed_answers"].append(record)
        
        # 질문 패턴 학습
        domain = self._detect_domain(question)
        if domain not in self.learning_data["question_patterns"]:
            self.learning_data["question_patterns"][domain] = {"count": 0, "avg_quality": 0.0}
        
        patterns = self.learning_data["question_patterns"][domain]
        patterns["count"] += 1
        patterns["avg_quality"] = (patterns["avg_quality"] * (patterns["count"] - 1) + quality_score) / patterns["count"]
        
        # 품질 점수 기록
        self.learning_data["answer_quality_scores"].append(quality_score)
    
    def _calculate_korean_ratio(self, text: str) -> float:
        """한국어 비율 계산"""
        if not text:
            return 0.0
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.sub(r'[^\w가-힣]', '', text))
        
        if total_chars == 0:
            return 0.0
        
        return korean_chars / total_chars
    
    def _warmup(self):
        """모델 워밍업"""
        try:
            test_prompt = "테스트"
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False
                )
            if self.verbose:
                print("모델 워밍업 완료")
        except Exception as e:
            if self.verbose:
                print(f"워밍업 실패: {e}")
    
    def generate_answer(self, question: str, question_type: str) -> str:
        """답변 생성"""
        
        # 프롬프트 생성
        if question_type == "multiple_choice":
            prompt = self._create_mc_prompt(question)
        else:
            prompt = self._create_korean_subj_prompt(question)
        
        try:
            # 토크나이징
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            )
            
            if self.device == "cuda":
                inputs = inputs.to(self.model.device)
            
            # 생성 설정
            gen_config = self._get_generation_config(question_type)
            
            # 모델 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            
            # 디코딩
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # 후처리
            if question_type == "multiple_choice":
                answer = self._process_mc_answer(response)
                self._add_learning_record(question, answer, question_type, True, 1.0)
                return answer
            else:
                answer = self._process_korean_subj_answer(response, question)
                korean_ratio = self._calculate_korean_ratio(answer)
                quality_score = self._calculate_answer_quality(answer, question)
                success = korean_ratio > 0.7 and quality_score > 0.5
                
                self._add_learning_record(question, answer, question_type, success, quality_score)
                return answer
                
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            fallback = self._get_fallback_answer(question_type, question)
            self._add_learning_record(question, fallback, question_type, False, 0.3)
            return fallback
    
    def _create_mc_prompt(self, question: str) -> str:
        """객관식 프롬프트 생성"""
        prompts = [
            f"""다음은 금융보안 관련 문제입니다. 정답을 선택하세요.

{question}

위 문제의 정답은 1, 2, 3, 4, 5 중 하나입니다.
정답 번호만 답하세요.

정답:""",
            
            f"""금융보안 전문가로서 다음 문제를 해결하세요.

{question}

선택지 중 가장 적절한 답을 1, 2, 3, 4, 5 중에서 선택하세요.
번호만 답하세요.

답:""",
            
            f"""다음 금융보안 문제를 분석하고 정답을 선택하세요.

문제: {question}

정답을 1, 2, 3, 4, 5 중 하나의 번호로만 답하세요.

정답:"""
        ]
        
        return random.choice(prompts)
    
    def _create_korean_subj_prompt(self, question: str) -> str:
        """한국어 전용 주관식 프롬프트 생성"""
        domain = self._detect_domain(question)
        
        prompts = [
            f"""금융보안 전문가로서 다음 질문에 대해 한국어로만 정확한 답변을 작성하세요.

질문: {question}

답변 작성 시 다음 사항을 준수하세요:
- 반드시 한국어로만 작성
- 관련 법령과 규정을 근거로 구체적 내용 포함
- 실무적이고 전문적인 관점에서 설명
- 영어 용어 사용 금지

답변:""",
            
            f"""다음은 {domain} 분야의 전문 질문입니다. 한국어로만 상세하고 정확한 답변을 제공하세요.

{question}

한국어 전용 답변 작성 기준:
- 모든 전문 용어를 한국어로 표기
- 법적 근거와 실무 절차를 한국어로 설명
- 영어나 외국어 사용 금지

답변:""",
            
            f"""{domain} 전문가의 관점에서 다음 질문에 한국어로만 답변하세요.

질문: {question}

답변 요구사항:
- 완전한 한국어 답변
- 관련 법령과 규정을 한국어로 설명
- 체계적이고 구체적인 한국어 설명

답변:"""
        ]
        
        return random.choice(prompts)
    
    def _get_generation_config(self, question_type: str) -> GenerationConfig:
        """생성 설정"""
        if question_type == "multiple_choice":
            return GenerationConfig(
                max_new_tokens=15,
                temperature=0.3,
                top_p=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            return GenerationConfig(
                max_new_tokens=300,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
    
    def _process_mc_answer(self, response: str) -> str:
        """객관식 답변 처리"""
        # 숫자 추출
        numbers = re.findall(r'[1-5]', response)
        if numbers:
            answer = numbers[0]
            self.answer_distribution[answer] += 1
            self.total_mc_answers += 1
            return answer
        
        # 폴백: 분포 균등화
        return self._get_balanced_mc_answer()
    
    def _process_korean_subj_answer(self, response: str, question: str) -> str:
        """한국어 전용 주관식 답변 처리"""
        # 기본 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 영어와 중국어 제거
        response = re.sub(r'[a-zA-Z]+', '', response)  # 영어 제거
        response = re.sub(r'[\u4e00-\u9fff]+', '', response)  # 중국어 제거
        response = re.sub(r'[①②③④⑤➀➁➂➃➄]', '', response)  # 특수 기호 제거
        
        # 불필요한 문자 정리
        response = re.sub(r'[^\w\s가-힣.,!?()[\]-]', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 한국어 비율 확인
        korean_ratio = self._calculate_korean_ratio(response)
        
        # 한국어 비율이 낮거나 길이가 짧으면 템플릿 사용
        if korean_ratio < 0.8 or len(response) < 30:
            return self._generate_korean_template_answer(question)
        
        # 길이 제한
        if len(response) > 350:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3])
            if not response.endswith('.'):
                response += '.'
        
        # 마침표 확인
        if not response.endswith(('.', '다', '요', '함')):
            response += '.'
        
        return response
    
    def _generate_korean_template_answer(self, question: str) -> str:
        """한국어 템플릿 답변 생성"""
        domain = self._detect_domain(question)
        
        # 도메인별 템플릿 사용
        if domain in self.korean_templates:
            templates = self.korean_templates[domain]
        else:
            # 일반 템플릿
            templates = [
                "관련 법령과 규정에 따라 체계적인 관리 방안을 수립하고 지속적인 모니터링을 수행해야 합니다.",
                "전문적인 보안 정책을 수립하고 정기적인 점검과 평가를 실시하여 보안 수준을 유지해야 합니다.", 
                "법적 요구사항을 준수하며 효과적인 보안 조치를 시행하고 관련 교육을 실시해야 합니다.",
                "위험 요소를 식별하고 적절한 대응 방안을 마련하여 체계적으로 관리해야 합니다.",
                "관리 절차를 확립하고 정기적인 평가를 통해 지속적인 개선을 추진해야 합니다."
            ]
        
        return random.choice(templates)
    
    def _calculate_answer_quality(self, answer: str, question: str) -> float:
        """답변 품질 점수 계산"""
        if not answer:
            return 0.0
        
        score = 0.0
        
        # 한국어 비율 (40%)
        korean_ratio = self._calculate_korean_ratio(answer)
        score += korean_ratio * 0.4
        
        # 길이 적절성 (20%)
        length = len(answer)
        if 50 <= length <= 400:
            score += 0.2
        elif 30 <= length < 50 or 400 < length <= 500:
            score += 0.1
        
        # 문장 구조 (20%)
        if answer.endswith(('.', '다', '요', '함')):
            score += 0.1
        
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.1
        
        # 전문성 (20%)
        domain_keywords = self._get_domain_keywords(question)
        found_keywords = sum(1 for keyword in domain_keywords if keyword in answer)
        if found_keywords > 0:
            score += min(found_keywords / len(domain_keywords), 1.0) * 0.2
        
        return min(score, 1.0)
    
    def _get_domain_keywords(self, question: str) -> List[str]:
        """도메인별 키워드 반환"""
        question_lower = question.lower()
        
        if "개인정보" in question_lower:
            return ["개인정보보호법", "정보주체", "처리", "보호조치", "동의"]
        elif "전자금융" in question_lower:
            return ["전자금융거래법", "접근매체", "인증", "보안", "분쟁조정"]
        elif "보안" in question_lower or "악성코드" in question_lower:
            return ["보안정책", "탐지", "대응", "모니터링", "방어"]
        else:
            return ["법령", "규정", "관리", "조치", "절차"]
    
    def _get_balanced_mc_answer(self) -> str:
        """균등 분포 객관식 답변"""
        if self.total_mc_answers < 5:
            return str(random.randint(1, 5))
        
        # 현재 분포 확인
        avg_count = self.total_mc_answers / 5
        underused = [num for num in ["1", "2", "3", "4", "5"] 
                    if self.answer_distribution[num] < avg_count * 0.7]
        
        if underused:
            answer = random.choice(underused)
        else:
            answer = str(random.randint(1, 5))
        
        self.answer_distribution[answer] += 1
        self.total_mc_answers += 1
        return answer
    
    def _get_fallback_answer(self, question_type: str, question: str = "") -> str:
        """폴백 답변"""
        if question_type == "multiple_choice":
            return self._get_balanced_mc_answer()
        else:
            return self._generate_korean_template_answer(question)
    
    def _detect_domain(self, question: str) -> str:
        """도메인 감지"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["개인정보", "정보주체"]):
            return "개인정보보호"
        elif any(word in question_lower for word in ["전자금융", "전자적"]):
            return "전자금융"
        elif any(word in question_lower for word in ["보안", "악성코드", "트로이"]):
            return "사이버보안"
        elif any(word in question_lower for word in ["정보보안", "isms"]):
            return "정보보안"
        elif any(word in question_lower for word in ["금융투자", "자본시장"]):
            return "금융투자"
        elif any(word in question_lower for word in ["위험관리", "리스크"]):
            return "위험관리"
        else:
            return "일반"
    
    def get_answer_stats(self) -> Dict:
        """답변 통계"""
        return {
            "distribution": dict(self.answer_distribution),
            "total_mc": self.total_mc_answers
        }
    
    def get_learning_stats(self) -> Dict:
        """학습 통계"""
        return {
            "successful_count": len(self.learning_data["successful_answers"]),
            "failed_count": len(self.learning_data["failed_answers"]),
            "question_patterns": dict(self.learning_data["question_patterns"]),
            "avg_quality": sum(self.learning_data["answer_quality_scores"]) / len(self.learning_data["answer_quality_scores"]) if self.learning_data["answer_quality_scores"] else 0
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 학습 데이터 저장
            self._save_learning_data()
            
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            if self.verbose:
                print(f"정리 중 오류: {e}")