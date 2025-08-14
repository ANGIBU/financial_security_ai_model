# model_handler.py

"""
LLM 모델 핸들러
- 모델 로딩 및 관리
- 답변 생성
- 프롬프트 처리
"""

import torch
import re
import time
import gc
import random
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
        
        # 주관식 답변 템플릿
        self.subj_templates = [
            "관련 법령에 따라 {action}하고 {process}를 수행해야 합니다.",
            "{domain} 분야에서는 {requirement}을 준수하며 {measure}를 시행해야 합니다.",
            "체계적인 {system}을 구축하고 지속적인 {monitoring}을 실시해야 합니다.",
            "{standard}에 따른 {procedure}를 수립하고 정기적인 {evaluation}을 수행해야 합니다.",
            "적절한 {control}을 구현하고 효과적인 {management}를 통해 관리해야 합니다."
        ]
        
        # 워밍업
        self._warmup()
        
        print("모델 로딩 완료")
    
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
            prompt = self._create_subj_prompt(question)
        
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
                return self._process_mc_answer(response)
            else:
                return self._process_subj_answer(response, question)
                
        except Exception as e:
            if self.verbose:
                print(f"모델 실행 오류: {e}")
            return self._get_fallback_answer(question_type)
    
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
    
    def _create_subj_prompt(self, question: str) -> str:
        """주관식 프롬프트 생성"""
        domain = self._detect_domain(question)
        
        prompts = [
            f"""금융보안 전문가로서 다음 질문에 대해 전문적이고 정확한 답변을 작성하세요.

질문: {question}

답변은 한국어로 작성하고, 관련 법령과 규정을 근거로 구체적이고 실무적인 내용을 포함하세요.

답변:""",
            
            f"""다음은 {domain} 분야의 전문 질문입니다. 상세하고 정확한 답변을 제공하세요.

{question}

전문적인 관점에서 법적 근거와 실무적 방안을 포함하여 한국어로 답변하세요.

답변:""",
            
            f"""{domain} 전문가의 관점에서 다음 질문에 답변하세요.

질문: {question}

관련 법령, 규정, 실무 절차를 포함하여 체계적이고 구체적으로 한국어로 설명하세요.

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
                max_new_tokens=350,
                temperature=0.7,
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
    
    def _process_subj_answer(self, response: str, question: str) -> str:
        """주관식 답변 처리"""
        # 기본 정리
        response = re.sub(r'\s+', ' ', response).strip()
        
        # 중국어 제거
        response = re.sub(r'[\u4e00-\u9fff]+', '', response)
        
        # 길이 확인
        if len(response) < 20:
            return self._generate_subj_fallback(question)
        
        # 길이 제한
        if len(response) > 400:
            sentences = response.split('. ')
            response = '. '.join(sentences[:3])
            if not response.endswith('.'):
                response += '.'
        
        # 마침표 확인
        if not response.endswith(('.', '다', '요')):
            response += '.'
        
        return response
    
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
    
    def _generate_subj_fallback(self, question: str) -> str:
        """주관식 폴백 답변 생성"""
        domain = self._detect_domain(question)
        
        if "개인정보" in question:
            templates = [
                "개인정보보호법에 따라 정보주체의 권리를 보장하고 적절한 보호조치를 시행해야 합니다.",
                "관련 법령에 따른 개인정보 처리 원칙을 준수하고 체계적인 관리방안을 수립해야 합니다.",
                "정보주체의 동의를 받고 목적 범위 내에서 처리하며 안전성 확보조치를 이행해야 합니다."
            ]
        elif "전자금융" in question:
            templates = [
                "전자금융거래법에 따른 보안 기준을 충족하고 이용자 보호를 위한 조치를 시행해야 합니다.",
                "금융감독원 또는 한국은행에서 관련 업무를 담당하며 법정 절차를 준수해야 합니다.",
                "전자금융분쟁조정위원회를 통해 분쟁조정을 신청할 수 있으며 관련 규정을 따라야 합니다."
            ]
        elif "보안" in question or "악성코드" in question:
            templates = [
                "다층 보안체계를 구축하고 지속적인 모니터링을 통해 위협을 탐지하고 대응해야 합니다.",
                "보안정책을 수립하고 정기적인 점검과 교육을 통해 보안 수준을 유지해야 합니다.",
                "기술적, 관리적, 물리적 보안조치를 종합적으로 적용하여 체계적으로 관리해야 합니다."
            ]
        else:
            templates = self.subj_templates
        
        template = random.choice(templates)
        
        # 템플릿 변수 채우기
        replacements = {
            "action": random.choice(["보안조치를 수립", "관리체계를 구축", "정책을 마련"]),
            "process": random.choice(["지속적인 모니터링", "정기적인 평가", "체계적인 관리"]),
            "domain": domain,
            "requirement": random.choice(["법적 요구사항", "보안 기준", "관리 원칙"]),
            "measure": random.choice(["적절한 조치", "보안 대책", "관리 방안"]),
            "system": random.choice(["보안 관리체계", "내부 통제시스템", "위험 관리체계"]),
            "monitoring": random.choice(["점검", "모니터링", "평가"]),
            "standard": random.choice(["관련 법령", "보안 표준", "관리 기준"]),
            "procedure": random.choice(["절차", "정책", "방안"]),
            "evaluation": random.choice(["검토", "평가", "점검"]),
            "control": random.choice(["보안통제", "관리조치", "보안조치"]),
            "management": random.choice(["관리", "운영", "통제"])
        }
        
        for key, value in replacements.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template
    
    def _detect_domain(self, question: str) -> str:
        """도메인 감지"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["개인정보", "정보주체"]):
            return "개인정보보호"
        elif any(word in question_lower for word in ["전자금융", "전자적"]):
            return "전자금융"
        elif any(word in question_lower for word in ["보안", "악성코드", "트로이"]):
            return "보안"
        else:
            return "금융"
    
    def _get_fallback_answer(self, question_type: str) -> str:
        """폴백 답변"""
        if question_type == "multiple_choice":
            return self._get_balanced_mc_answer()
        else:
            return "관련 법령과 규정에 따라 적절한 보안 조치를 수립하고 체계적으로 관리해야 합니다."
    
    def get_answer_stats(self) -> Dict:
        """답변 통계"""
        return {
            "distribution": dict(self.answer_distribution),
            "total_mc": self.total_mc_answers
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
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