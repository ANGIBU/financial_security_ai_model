# prompt_engineering.py

"""
프롬프트 엔지니어링
- 객관식/주관식 프롬프트 생성
- 도메인별 템플릿 관리
- 한국어 강화 프롬프트
- 패턴 기반 힌트 적용
"""

import gc
import hashlib
import re
from typing import Dict, List, Optional, Tuple

from knowledge_base import FinancialSecurityKnowledgeBase

# 상수 정의
DEFAULT_CACHE_SIZE = 200
CACHE_CLEANUP_INTERVAL = 50
PROMPT_MAX_LENGTH = 2000
TEMPLATE_CACHE_SIZE = 20

class PromptEngineer:
    
    def __init__(self):
        """프롬프트 엔지니어링 시스템 초기화"""
        try:
            self.knowledge_base = FinancialSecurityKnowledgeBase()
            self.templates = self._build_optimized_templates()
            
            # 캐시 시스템
            self.prompt_cache = {}
            self.template_cache = {}
            self.max_cache_size = DEFAULT_CACHE_SIZE
            self.cache_cleanup_counter = 0
            
            # 통계 추적
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "template_usage": {},
                "domain_distribution": {},
                "prompt_generations": 0,
                "avg_prompt_length": 0.0
            }
            
        except Exception as e:
            raise RuntimeError(f"프롬프트 엔지니어 초기화 실패: {e}")
    
    def _build_optimized_templates(self) -> Dict[str, str]:
        """최적화된 템플릿 구축"""
        templates = {}
        
        # 객관식 기본 템플릿
        templates["mc_basic"] = """{question}

위 문제의 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 객관식 부정형 템플릿
        templates["mc_negative"] = """{question}

이 문제는 틀린 것, 해당하지 않는 것, 적절하지 않은 것을 찾는 문제입니다.
정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 금융투자업 특화 템플릿
        templates["mc_financial"] = """{question}

금융투자업 분류:
- 금융투자업: 투자매매업, 투자중개업, 투자자문업, 투자일임업
- 금융투자업 아님: 소비자금융업, 보험중개업

정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 주관식 기본 템플릿
        templates["subj_basic"] = """{question}

위 질문에 대해 한국어로 답변하세요.
법령과 규정에 따른 구체적인 설명을 포함하세요."""

        # 사이버보안 특화 템플릿
        templates["subj_trojan"] = """{question}

트로이 목마의 특징과 탐지 방법에 대해 한국어로 설명하세요."""

        # 개인정보보호 특화 템플릿
        templates["subj_personal_info"] = """{question}

개인정보보호법에 따른 조치사항을 한국어로 설명하세요."""

        # 전자금융 특화 템플릿
        templates["subj_electronic"] = """{question}

전자금융거래법에 따른 안전성 확보 방안을 한국어로 설명하세요."""

        # 정보보안 특화 템플릿
        templates["subj_info_security"] = """{question}

정보보안 관리체계에 따른 체계적인 관리 방안을 한국어로 설명하세요."""

        # 위험관리 특화 템플릿
        templates["subj_risk_management"] = """{question}

위험관리 체계 구축 방안을 한국어로 설명하세요."""

        return templates
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        """한국어 강화 프롬프트 생성"""
        if not question or not question.strip():
            raise ValueError("질문이 비어있습니다")
        
        if len(question) > PROMPT_MAX_LENGTH:
            question = question[:PROMPT_MAX_LENGTH-3] + "..."
        
        # 캐시 키 생성 (충돌 방지를 위한 개선)
        cache_content = f"{question[:200]}{question_type}{len(question)}"
        cache_key = hashlib.sha256(cache_content.encode('utf-8')).hexdigest()[:16]
        
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        try:
            question_lower = question.lower()
            
            # 템플릿 선택 로직
            if question_type == "multiple_choice":
                template_key = self._select_mc_template(question_lower)
            else:
                template_key = self._select_subj_template(question_lower)
            
            # 템플릿 적용
            if template_key not in self.templates:
                template_key = "mc_basic" if question_type == "multiple_choice" else "subj_basic"
            
            prompt = self.templates[template_key].format(question=question.strip())
            
            # 통계 업데이트
            self.stats["template_usage"][template_key] = self.stats["template_usage"].get(template_key, 0) + 1
            self.stats["prompt_generations"] += 1
            
            # 평균 프롬프트 길이 업데이트
            total_length = self.stats["avg_prompt_length"] * (self.stats["prompt_generations"] - 1)
            total_length += len(prompt)
            self.stats["avg_prompt_length"] = total_length / self.stats["prompt_generations"]
            
            # 캐시 관리
            self._manage_cache()
            self.prompt_cache[cache_key] = prompt
            
            return prompt
            
        except Exception as e:
            # 오류 발생 시 기본 템플릿 사용
            fallback_key = "mc_basic" if question_type == "multiple_choice" else "subj_basic"
            return self.templates[fallback_key].format(question=question.strip())
    
    def _select_mc_template(self, question_lower: str) -> str:
        """객관식 템플릿 선택"""
        # 금융투자업 관련 특수 처리
        if ("금융투자업" in question_lower and 
            ("소비자금융업" in question_lower or "보험중개업" in question_lower)):
            return "mc_financial"
        
        # 부정형 질문 처리
        negative_patterns = ["해당하지", "적절하지", "옳지", "틀린", "잘못된", "부적절한"]
        if any(pattern in question_lower for pattern in negative_patterns):
            return "mc_negative"
        
        return "mc_basic"
    
    def _select_subj_template(self, question_lower: str) -> str:
        """주관식 템플릿 선택"""
        # 도메인별 특화 템플릿 선택
        domain_keywords = {
            "subj_trojan": ["트로이", "악성코드", "멀웨어", "바이러스"],
            "subj_personal_info": ["개인정보", "정보주체", "개인정보보호법"],
            "subj_electronic": ["전자금융", "전자적", "접근매체", "전자금융거래법"],
            "subj_info_security": ["정보보안", "보안관리", "ISMS", "보안정책"],
            "subj_risk_management": ["위험관리", "위험평가", "위험분석", "위험통제"]
        }
        
        for template_key, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return template_key
        
        return "subj_basic"
    
    def _manage_cache(self) -> None:
        """캐시 관리"""
        self.cache_cleanup_counter += 1
        
        if self.cache_cleanup_counter % CACHE_CLEANUP_INTERVAL == 0:
            # 프롬프트 캐시 정리
            if len(self.prompt_cache) >= self.max_cache_size:
                keys_to_remove = list(self.prompt_cache.keys())[:self.max_cache_size // 3]
                for key in keys_to_remove:
                    del self.prompt_cache[key]
            
            # 템플릿 캐시 정리
            if len(self.template_cache) >= TEMPLATE_CACHE_SIZE:
                keys_to_remove = list(self.template_cache.keys())[:TEMPLATE_CACHE_SIZE // 2]
                for key in keys_to_remove:
                    del self.template_cache[key]
            
            # 메모리 정리
            gc.collect()
    
    def create_prompt(self, question: str, question_type: str, 
                     analysis: Optional[Dict] = None, 
                     structure: Optional[Dict] = None) -> str:
        """범용 프롬프트 생성 (호환성 유지)"""
        return self.create_korean_reinforced_prompt(question, question_type)
    
    def create_few_shot_prompt(self, question: str, question_type: str, 
                              analysis: Optional[Dict] = None, 
                              num_examples: int = 1) -> str:
        """Few-shot 프롬프트 생성"""
        try:
            prompt_parts = []
            
            if question_type == "multiple_choice":
                # 객관식 예시
                prompt_parts.append("예시: 개인정보의 정의로 가장 적절한 것은?")
                prompt_parts.append("정답: 2")
                prompt_parts.append("")
            else:
                # 주관식 예시
                prompt_parts.append("예시: 트로이 목마의 특징을 설명하세요.")
                prompt_parts.append("답변: 트로이 목마는 정상 프로그램으로 위장한 악성코드입니다.")
                prompt_parts.append("")
            
            prompt_parts.append(f"문제: {question}")
            
            if question_type == "multiple_choice":
                prompt_parts.append("정답:")
            else:
                prompt_parts.append("답변:")
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            # 오류 시 기본 프롬프트로 폴백
            return self.create_korean_reinforced_prompt(question, question_type)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        """모델별 프롬프트 최적화"""
        if not prompt or not model_name:
            return prompt
        
        try:
            model_name_lower = model_name.lower()
            
            # SOLAR 모델 최적화
            if "solar" in model_name_lower:
                return f"### User:\n{prompt}\n\n### Assistant:\n"
            
            # LLaMA 모델 최적화
            elif "llama" in model_name_lower:
                return f"[INST] {prompt} [/INST]"
            
            # Mistral 모델 최적화
            elif "mistral" in model_name_lower:
                return f"<s>[INST] {prompt} [/INST]"
            
            # 기본 포맷
            else:
                return prompt
                
        except Exception:
            return prompt
    
    def get_template_suggestions(self, question: str) -> List[Tuple[str, float]]:
        """질문에 적합한 템플릿 제안"""
        suggestions = []
        question_lower = question.lower()
        
        try:
            # 각 템플릿에 대한 적합도 계산
            template_scores = {}
            
            for template_key in self.templates.keys():
                score = self._calculate_template_fitness(question_lower, template_key)
                if score > 0.1:
                    template_scores[template_key] = score
            
            # 점수 순으로 정렬
            suggestions = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
            
        except Exception:
            # 오류 시 기본 제안
            if "multiple_choice" in question_lower or any(str(i) in question for i in range(1, 6)):
                suggestions = [("mc_basic", 0.5)]
            else:
                suggestions = [("subj_basic", 0.5)]
        
        return suggestions[:3]
    
    def _calculate_template_fitness(self, question_lower: str, template_key: str) -> float:
        """템플릿 적합도 계산"""
        score = 0.0
        
        fitness_keywords = {
            "mc_basic": ["선택", "번호", "정답"],
            "mc_negative": ["해당하지", "적절하지", "옳지", "틀린"],
            "mc_financial": ["금융투자업", "소비자금융업", "보험중개업"],
            "subj_trojan": ["트로이", "악성코드", "멀웨어"],
            "subj_personal_info": ["개인정보", "정보주체"],
            "subj_electronic": ["전자금융", "접근매체"],
            "subj_info_security": ["정보보안", "보안관리"],
            "subj_risk_management": ["위험관리", "위험평가"]
        }
        
        keywords = fitness_keywords.get(template_key, [])
        for keyword in keywords:
            if keyword in question_lower:
                score += 1.0 / len(keywords)
        
        return min(score, 1.0)
    
    def get_stats_report(self) -> Dict:
        """통계 보고서 생성"""
        total_prompts = self.stats["prompt_generations"]
        cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        
        report = {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(cache_requests, 1),
            "cache_size": len(self.prompt_cache),
            "template_usage": dict(self.stats["template_usage"]),
            "domain_distribution": dict(self.stats["domain_distribution"]),
            "avg_prompt_length": self.stats["avg_prompt_length"],
            "cache_efficiency": {
                "prompt_cache_size": len(self.prompt_cache),
                "template_cache_size": len(self.template_cache),
                "cleanup_count": self.cache_cleanup_counter
            }
        }
        
        # 가장 많이 사용된 템플릿
        if self.stats["template_usage"]:
            most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
            report["most_used_template"] = {"name": most_used[0], "count": most_used[1]}
        
        return report
    
    def validate_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """프롬프트 유효성 검증"""
        issues = []
        
        if not prompt:
            issues.append("프롬프트가 비어있음")
            return False, issues
        
        if len(prompt) < 10:
            issues.append("프롬프트가 너무 짧음")
        
        if len(prompt) > PROMPT_MAX_LENGTH:
            issues.append(f"프롬프트가 너무 김 (최대 {PROMPT_MAX_LENGTH}자)")
        
        # 한국어 비율 확인
        korean_chars = len([c for c in prompt if '가' <= c <= '힣'])
        total_chars = len([c for c in prompt if c.isalnum()])
        
        if total_chars > 0:
            korean_ratio = korean_chars / total_chars
            if korean_ratio < 0.3:
                issues.append("한국어 비율이 낮음")
        
        # 문제 문자 확인
        if re.search(r'[\u4e00-\u9fff]', prompt):
            issues.append("중국어 문자 포함")
        
        return len(issues) == 0, issues
    
    def cleanup(self) -> None:
        """리소스 정리"""
        try:
            # 통계 요약
            total_usage = sum(self.stats["template_usage"].values())
            if total_usage > 0:
                most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1])
                cache_efficiency = self.stats["cache_hits"] / max(
                    self.stats["cache_hits"] + self.stats["cache_misses"], 1
                )
            
            # 캐시 정리
            self.prompt_cache.clear()
            self.template_cache.clear()
            
            # 통계 정리
            self.stats = {
                "cache_hits": 0,
                "cache_misses": 0,
                "template_usage": {},
                "domain_distribution": {},
                "prompt_generations": 0,
                "avg_prompt_length": 0.0
            }
            
            # 메모리 정리
            gc.collect()
            
        except Exception as e:
            # 정리 중 오류가 발생해도 계속 진행
            pass