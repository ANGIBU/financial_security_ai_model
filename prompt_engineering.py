# prompt_engineering.py
"""
프롬프트 엔지니어링 시스템
"""

import re
import hashlib
from typing import Dict, List, Optional
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    """프롬프트 엔지니어링 클래스"""
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_korean_templates()
        self.examples = self._build_korean_examples()
        
        self.prompt_cache = {}
        self.template_cache = {}
        self.max_cache_size = 150
        
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {}
        }
    
    def _build_korean_templates(self) -> Dict[str, str]:
        """한국어 전용 템플릿"""
        templates = {}
        
        templates["mc_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 절대 규칙
1. 오직 1, 2, 3, 4, 5 중 하나의 숫자만 답변하세요
2. 분석이나 설명 없이 숫자만 출력하세요
3. 반드시 한 개의 정답만 선택하세요

### 문제
{question}

정답:"""

        templates["mc_negative"] = """당신은 한국의 금융보안 전문가입니다.

### 절대 규칙
1. 이 문제는 '{keyword}' 문제입니다
2. 틀린 것 또는 해당하지 않는 것을 찾으세요
3. 반드시 1, 2, 3, 4, 5 중 하나만 답하세요

### 문제
{question}

### 부정형 문제 힌트
- "해당하지 않는" = 맞지 않는 것을 찾기
- "적절하지 않은" = 옳지 않은 것을 찾기
- "틀린 것" = 잘못된 것을 찾기

정답:"""

        templates["mc_financial_basic"] = """당신은 한국의 금융보안 전문가입니다.

### 절대 규칙
1. 반드시 1, 2, 3, 4, 5 중 하나만 답하세요
2. 금융업무 분류 관련 문제입니다
3. 정확한 금융업 구분을 적용하세요

### 문제
{question}

### 금융업 힌트
- 금융투자업: 투자매매업, 투자중개업, 투자자문업, 투자일임업
- 보험업: 보험중개업, 보험대리점업
- 소비자금융업: 별도 분류

정답:"""

        templates["subj_enhanced"] = """당신은 한국의 금융보안 전문가입니다.

### 매우 중요한 규칙
1. 반드시 순수 한국어로만 답변하세요
2. 한자, 영어, 일본어, 중국어 등 모든 외국어 절대 금지
3. 100-250자 내외로 답변하세요
4. 전문적이고 명확한 한국어만 사용

### 질문
{question}

### 답변 형식
관련 법령과 규정에 따라 구체적으로 설명하되,
순수 한국어만 사용하여 답변하세요.

답변:"""

        templates["subj_trojan"] = """당신은 한국의 사이버보안 전문가입니다.

### 매우 중요한 규칙
1. 반드시 순수 한국어로만 답변하세요
2. 한자, 영어, 일본어, 중국어 등 모든 외국어 절대 금지
3. 120-220자 내외로 답변하세요
4. 트로이 목마와 원격 제어 악성코드에 대해 전문적으로 설명하세요

### 질문
{question}

### 답변 구조 (한국어로만)
1. 트로이 목마의 특징 설명
2. 원격 접근 트로이 목마의 기능
3. 주요 탐지 지표 나열

### 외국어 사용 절대 금지
- Trojan → 트로이 목마
- RAT → 원격 접근 트로이 목마
- Remote → 원격
- Access → 접근
- 모든 영어 단어 한국어로 변환 필수

답변:"""
        
        return templates
    
    def _build_korean_examples(self) -> Dict[str, Dict]:
        """한국어 예시"""
        examples = {
            "mc_financial": {
                "question": "금융투자업의 구분에 해당하지 않는 것은?",
                "answer": "1",
                "reasoning": "소비자금융업은 금융투자업이 아님"
            },
            "mc_risk": {
                "question": "위험 관리 계획 수립 시 고려해야 할 요소로 적절하지 않은 것은?",
                "answer": "2",
                "reasoning": "위험 수용은 대응 전략의 하나"
            },
            "subj_trojan": {
                "question": "트로이 목마 기반 원격제어 악성코드의 특징과 탐지 지표를 설명하세요.",
                "answer": "트로이 목마는 정상 프로그램으로 위장한 악성코드로, 원격 접근 트로이 목마는 공격자가 감염된 시스템을 원격으로 제어할 수 있게 합니다. 주요 탐지 지표로는 비정상적인 네트워크 연결, 시스템 리소스 사용 증가, 알 수 없는 프로세스 실행, 방화벽 규칙 변경 등이 있습니다."
            }
        }
        return examples
    
    def create_prompt(self, question: str, question_type: str, 
                     analysis: Dict, structure: Dict) -> str:
        """프롬프트 생성"""
        
        cache_key = hash(f"{question[:100]}{question_type}")
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        if question_type == "multiple_choice":
            prompt = self._create_mc_prompt_enhanced(question, analysis, structure)
        else:
            prompt = self._create_subj_prompt_enhanced(question, analysis, structure)
        
        if len(self.prompt_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.prompt_cache))
            del self.prompt_cache[oldest_key]
        self.prompt_cache[cache_key] = prompt
        
        self._update_stats(analysis)
        
        return prompt
    
    def _create_mc_prompt_enhanced(self, question: str, analysis: Dict, structure: Dict) -> str:
        """강화된 객관식 프롬프트 생성"""
        
        if "금융투자업" in question or "소비자금융업" in question:
            prompt = self.templates["mc_financial_basic"].format(question=question)
            self.stats["template_usage"]["mc_financial_basic"] = self.stats["template_usage"].get("mc_financial_basic", 0) + 1
            
        elif structure.get("has_negative", False):
            negative_keywords = ["해당하지 않는", "적절하지 않은", "옳지 않은", "틀린"]
            keyword = next((k for k in negative_keywords if k in question), "해당하지 않는")
            
            prompt = self.templates["mc_negative"].format(
                question=question,
                keyword=keyword
            )
            self.stats["template_usage"]["mc_negative"] = self.stats["template_usage"].get("mc_negative", 0) + 1
            
        else:
            prompt = self.templates["mc_basic"].format(question=question)
            self.stats["template_usage"]["mc_basic"] = self.stats["template_usage"].get("mc_basic", 0) + 1
        
        return prompt
    
    def _create_subj_prompt_enhanced(self, question: str, analysis: Dict, structure: Dict) -> str:
        """강화된 주관식 프롬프트 생성"""
        
        if "트로이" in question and ("RAT" in question or "원격" in question):
            prompt = self.templates["subj_trojan"].format(question=question)
            self.stats["template_usage"]["subj_trojan"] = self.stats["template_usage"].get("subj_trojan", 0) + 1
        else:
            prompt = self.templates["subj_enhanced"].format(question=question)
            self.stats["template_usage"]["subj_enhanced"] = self.stats["template_usage"].get("subj_enhanced", 0) + 1
        
        return prompt
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        """한국어 강화 프롬프트 생성"""
        
        if question_type == "multiple_choice":
            
            if "금융투자업" in question or "소비자금융업" in question:
                prompt = f"""### 중요: 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요 ###

### 문제
{question}

### 핵심 힌트
금융투자업에는 투자매매업, 투자중개업, 투자자문업, 투자일임업이 포함됩니다.
소비자금융업과 보험중개업은 금융투자업이 아닙니다.

정답:"""
                
            elif "위험" in question and "관리" in question:
                prompt = f"""### 중요: 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요 ###

### 문제
{question}

### 핵심 힌트
위험관리 계획 수립 시 고려 요소는 대상, 기간, 수행인력, 위험 대응 전략입니다.
위험 수용은 위험 대응 전략의 하나이지 별도 고려 요소가 아닙니다.

정답:"""
                
            elif "관리체계" in question and "정책" in question:
                prompt = f"""### 중요: 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요 ###

### 문제
{question}

### 핵심 힌트
관리체계 수립의 정책 수립 단계에서 가장 중요한 것은 경영진의 참여와 지원입니다.
최고책임자 지정과 자원 할당은 그 다음 단계입니다.

정답:"""
                
            elif "재해" in question and "복구" in question:
                prompt = f"""### 중요: 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요 ###

### 문제
{question}

### 핵심 힌트
재해복구 계획에는 복구 절차, 비상연락체계, 복구 목표시간이 포함됩니다.
개인정보 파기 절차는 재해복구와 직접적인 관련이 없습니다.

정답:"""
                
            else:
                prompt = f"""### 중요: 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 답하세요 ###

### 문제
{question}

### 규칙
1. 숫자 하나만 선택하세요
2. 설명이나 분석은 하지 마세요
3. 정답만 제시하세요

정답:"""
        else:
            if "트로이" in question and ("RAT" in question or "원격" in question or "악성코드" in question):
                prompt = f"""### 매우 중요: 반드시 순수 한국어로만 답변하세요. 한자, 영어, 일본어 등 모든 외국어 절대 금지 ###

### 질문
{question}

### 답변 지침 (매우 중요)
1. 트로이 목마의 특징을 먼저 설명 (한국어로만)
2. 원격 접근 트로이 목마의 기능 설명 (한국어로만)
3. 주요 탐지 지표를 나열 (한국어로만)
4. 120-200자 내외로 작성
5. 절대로 외국어 사용 금지

### 외국어 변환 필수
- Trojan → 트로이 목마
- RAT → 원격 접근 트로이 목마  
- Remote → 원격
- Access → 접근
- Control → 제어
- Malware → 악성코드

답변:"""
            else:
                prompt = f"""### 매우 중요: 반드시 순수 한국어로만 답변하세요. 한자, 영어 등 모든 외국어 절대 금지 ###

### 질문
{question}

### 답변 규칙
1. 순수 한국어만 사용
2. 80자 이상 200자 이내
3. 관련 법령과 규정 언급
4. 구체적 방안 제시
5. 절대로 외국어 사용 금지

답변:"""
        
        return prompt
    
    def create_few_shot_prompt(self, question: str, question_type: str,
                             analysis: Dict, num_examples: int = 1) -> str:
        """Few-shot 프롬프트 생성"""
        
        prompt_parts = ["다음은 한국어 금융보안 문제 예시입니다.\n"]
        
        if question_type == "multiple_choice":
            if "금융투자업" in question:
                example = self.examples["mc_financial"]
            elif "위험" in question and "관리" in question:
                example = self.examples["mc_risk"]
            else:
                example = self.examples["mc_financial"]
            
            prompt_parts.append(f"예시 문제: {example['question']}")
            prompt_parts.append(f"정답: {example['answer']}\n")
        else:
            if "트로이" in question:
                example = self.examples["subj_trojan"]
                prompt_parts.append(f"예시 질문: {example['question']}")
                prompt_parts.append(f"답변: {example['answer']}\n")
        
        prompt_parts.append(f"현재 문제:\n{question}")
        prompt_parts.append("위 예시처럼 답하세요.")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 최적화"""
        
        korean_prefix = "### 중요: 반드시 한국어로만 답변하세요 ###\n\n"
        
        if "solar" in model_name.lower():
            optimized = f"{korean_prefix}### User:\n{prompt}\n\n### Assistant:\n"
        elif "llama" in model_name.lower():
            optimized = f"{korean_prefix}<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            optimized = korean_prefix + prompt
        
        return optimized
    
    def _update_stats(self, analysis: Dict):
        """통계 업데이트"""
        domains = analysis.get("domain", ["일반"])
        for domain in domains:
            self.stats["domain_distribution"][domain] = self.stats["domain_distribution"].get(domain, 0) + 1
    
    def get_stats_report(self) -> Dict:
        """통계 보고서"""
        total_prompts = sum(self.stats["template_usage"].values())
        
        return {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(total_prompts, 1),
            "template_usage": self.stats["template_usage"],
            "domain_distribution": self.stats["domain_distribution"]
        }
    
    def cleanup(self):
        """리소스 정리"""
        if self.stats["template_usage"]:
            total = sum(self.stats["template_usage"].values())
            if total > 0:
                most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
        
        self.prompt_cache.clear()
        self.template_cache.clear()