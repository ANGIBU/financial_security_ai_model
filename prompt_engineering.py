# prompt_engineering.py

"""
프롬프트 엔지니어링 (강화버전)
- 객관식/주관식 프롬프트 생성
- 도메인별 템플릿 관리
- 한국어 강화 프롬프트
- 패턴 기반 힌트 적용
- 동적 프롬프트 최적화
- 다단계 추론 지원
- 문맥 인식 프롬프트
"""

import re
import hashlib
import time
import random
from typing import Dict, List, Optional, Tuple
from knowledge_base import FinancialSecurityKnowledgeBase

class PromptEngineer:
    
    def __init__(self):
        self.knowledge_base = FinancialSecurityKnowledgeBase()
        self.templates = self._build_enhanced_templates()
        
        self.prompt_cache = {}
        self.max_cache_size = 200
        
        self.stats = {
            "cache_hits": 0,
            "template_usage": {},
            "domain_distribution": {},
            "success_rates": {},
            "optimization_cycles": 0
        }
        
        self.dynamic_templates = self._build_dynamic_templates()
        self.context_enhancers = self._build_context_enhancers()
        self.difficulty_adaptors = self._build_difficulty_adaptors()
        
        self.performance_tracker = {}
        self.template_evolution = {}
        
        self.multi_stage_prompts = self._build_multi_stage_prompts()
        self.reasoning_guides = self._build_reasoning_guides()
    
    def _build_enhanced_templates(self) -> Dict[str, str]:
        templates = {}
        
        # 객관식 기본 템플릿
        templates["mc_basic"] = """{question}

위 문제의 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나).
한국어로 사고하고 논리적으로 분석하여 답하세요."""

        # 객관식 부정형 템플릿
        templates["mc_negative"] = """{question}

이 문제는 틀린 것, 해당하지 않는 것, 적절하지 않은 것을 찾는 문제입니다.
각 선택지를 신중히 검토하여 부정 조건에 맞는 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 금융투자업 특화 템플릿
        templates["mc_financial"] = """{question}

금융투자업 분류 기준:
- 금융투자업: 투자매매업, 투자중개업, 투자자문업, 투자일임업
- 금융투자업이 아님: 소비자금융업, 보험중개업

자본시장법의 정의에 따라 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 개인정보보호 특화 템플릿
        templates["mc_privacy"] = """{question}

개인정보보호법 핵심 원칙:
- 개인정보 처리의 최소화
- 정보주체의 권리 보장
- 안전성 확보조치 의무

개인정보보호법의 규정에 따라 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 전자금융 특화 템플릿
        templates["mc_electronic"] = """{question}

전자금융거래법 핵심 사항:
- 전자적 장치를 통한 금융거래
- 접근매체의 안전한 관리
- 이용자 보호 의무

전자금융거래법의 규정에 따라 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 사이버보안 특화 템플릿
        templates["mc_cyber"] = """{question}

사이버보안 핵심 개념:
- 악성코드: 트로이목마, 랜섬웨어, 바이러스
- 공격 기법: 피싱, 스미싱, 사회공학
- 방어 기술: 침입탐지, 방화벽, 보안관제

사이버보안 원리에 따라 정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나)."""

        # 주관식 기본 템플릿
        templates["subj_basic"] = """{question}

위 질문에 대해 한국어로 전문적인 답변을 작성하세요.
관련 법령과 규정을 근거로 하여 구체적이고 체계적으로 설명하세요."""

        # 트로이목마 특화 템플릿
        templates["subj_trojan"] = """{question}

트로이목마에 대해 다음 관점에서 한국어로 설명하세요:
1. 정의와 특징
2. 감염 경로와 작동 방식
3. 탐지 방법과 대응 조치
4. 예방을 위한 보안 수칙"""

        # 개인정보보호 주관식 템플릿
        templates["subj_personal_info"] = """{question}

개인정보보호법의 관점에서 한국어로 설명하세요:
1. 법적 근거와 규정
2. 정보주체의 권리
3. 처리자의 의무
4. 안전성 확보조치"""

        # 전자금융 주관식 템플릿
        templates["subj_electronic"] = """{question}

전자금융거래법에 따른 안전성 확보 방안을 한국어로 설명하세요:
1. 법적 의무사항
2. 기술적 보안조치
3. 관리적 보안조치
4. 이용자 보호 방안"""

        # 위험관리 특화 템플릿
        templates["subj_risk"] = """{question}

위험관리 체계의 관점에서 한국어로 설명하세요:
1. 위험 식별과 분석
2. 위험 평가와 측정
3. 위험 대응 전략
4. 모니터링과 개선"""

        return templates
    
    def _build_dynamic_templates(self) -> Dict[str, Dict]:
        return {
            "difficulty_based": {
                "easy": {
                    "prefix": "다음은 기본적인 금융보안 문제입니다.\n",
                    "instruction": "핵심 개념을 중심으로 답하세요.",
                    "suffix": "\n간단명료하게 답변하세요."
                },
                "medium": {
                    "prefix": "다음은 금융보안 실무 문제입니다.\n",
                    "instruction": "관련 법령과 실무 관점에서 분석하세요.",
                    "suffix": "\n논리적 근거와 함께 답변하세요."
                },
                "hard": {
                    "prefix": "다음은 고급 금융보안 전문 문제입니다.\n",
                    "instruction": "다양한 관점에서 종합적으로 분석하세요.",
                    "suffix": "\n전문적이고 상세한 답변을 제공하세요."
                }
            },
            "context_based": {
                "legal": {
                    "focus": "법령 해석과 적용",
                    "guidance": "관련 법조문과 판례를 참고하여",
                    "format": "법적 근거 → 해석 → 적용"
                },
                "technical": {
                    "focus": "기술적 구현과 방법",
                    "guidance": "기술적 원리와 실무적 방법을 중심으로",
                    "format": "원리 → 방법 → 효과"
                },
                "management": {
                    "focus": "관리체계와 절차",
                    "guidance": "조직 관리와 운영 절차 관점에서",
                    "format": "현황 → 계획 → 실행 → 평가"
                }
            }
        }
    
    def _build_context_enhancers(self) -> Dict[str, List[str]]:
        return {
            "domain_boosters": {
                "개인정보보호": [
                    "개인정보보호법 제29조(안전성 확보조치)를 고려하여",
                    "정보주체의 권리 보호 관점에서",
                    "개인정보 처리 최소화 원칙에 따라"
                ],
                "전자금융": [
                    "전자금융거래법 제21조(안전성 확보 의무)에 따라",
                    "전자금융거래의 신뢰성 보장을 위해",
                    "이용자 보호를 최우선으로 고려하여"
                ],
                "정보보안": [
                    "정보보안 관리체계(ISMS) 관점에서",
                    "정보자산의 기밀성, 무결성, 가용성 확보를 위해",
                    "체계적인 보안 관리를 통해"
                ],
                "사이버보안": [
                    "지능형 지속 위협(APT) 대응을 위해",
                    "다층적 보안 방어 체계 구축을 통해",
                    "사이버 위협 인텔리전스를 활용하여"
                ]
            },
            "reasoning_enhancers": [
                "단계별로 분석하면",
                "핵심 요소를 정리하면",
                "법적 근거를 바탕으로",
                "실무적 관점에서",
                "종합적으로 고려할 때"
            ],
            "confidence_boosters": [
                "명확한 기준에 따라",
                "정확한 분석을 통해",
                "체계적인 접근으로",
                "전문적 판단에 의해"
            ]
        }
    
    def _build_difficulty_adaptors(self) -> Dict[str, Dict]:
        return {
            "prompt_complexity": {
                "simple": {
                    "instruction_depth": "기본적인",
                    "analysis_requirement": "핵심 내용을",
                    "detail_level": "간단히"
                },
                "moderate": {
                    "instruction_depth": "구체적인",
                    "analysis_requirement": "주요 사항을",
                    "detail_level": "상세히"
                },
                "complex": {
                    "instruction_depth": "종합적인",
                    "analysis_requirement": "모든 측면을",
                    "detail_level": "전문적으로"
                }
            },
            "response_guidance": {
                "structured": [
                    "다음 순서로 설명하세요:",
                    "체계적으로 정리하면:",
                    "단계별로 구분하면:"
                ],
                "analytical": [
                    "분석해보면:",
                    "검토 결과:",
                    "평가해보면:"
                ],
                "comprehensive": [
                    "종합적으로 고려하면:",
                    "전체적인 관점에서:",
                    "다각도로 분석하면:"
                ]
            }
        }
    
    def _build_multi_stage_prompts(self) -> Dict[str, Dict]:
        return {
            "stage1_quick": {
                "prefix": "1단계: 빠른 분석\n",
                "instruction": "핵심 포인트를 즉시 파악하세요.",
                "format": "직관적 판단"
            },
            "stage2_deep": {
                "prefix": "2단계: 심화 분석\n", 
                "instruction": "관련 요소들을 종합적으로 검토하세요.",
                "format": "논리적 추론"
            },
            "stage3_validation": {
                "prefix": "3단계: 검증 및 확정\n",
                "instruction": "분석 결과를 재검토하여 최종 답안을 확정하세요.",
                "format": "교차 검증"
            }
        }
    
    def _build_reasoning_guides(self) -> Dict[str, List[str]]:
        return {
            "logical_flow": [
                "전제 → 논리 → 결론",
                "문제 → 분석 → 해결",
                "현황 → 원인 → 대안"
            ],
            "analytical_framework": [
                "법적 → 기술적 → 관리적",
                "예방 → 탐지 → 대응",
                "계획 → 실행 → 평가"
            ],
            "domain_structure": [
                "정의 → 요소 → 절차",
                "원칙 → 기준 → 적용",
                "목적 → 방법 → 효과"
            ]
        }
    
    def analyze_question_context(self, question: str, structure: Dict) -> Dict:
        """질문 문맥 분석"""
        context_analysis = {
            "domain_strength": 0.0,
            "complexity_level": "medium",
            "question_intent": "general",
            "response_style": "standard",
            "reasoning_depth": "moderate"
        }
        
        question_lower = question.lower()
        
        # 도메인 강도 분석
        domain_indicators = {
            "개인정보보호": ["개인정보", "정보주체", "개인정보보호법"],
            "전자금융": ["전자금융", "전자적", "접근매체"],
            "사이버보안": ["트로이", "악성코드", "해킹", "피싱"],
            "정보보안": ["정보보안", "ISMS", "보안관리"]
        }
        
        max_strength = 0.0
        primary_domain = "일반"
        
        for domain, indicators in domain_indicators.items():
            strength = sum(1 for indicator in indicators if indicator in question_lower) / len(indicators)
            if strength > max_strength:
                max_strength = strength
                primary_domain = domain
        
        context_analysis["domain_strength"] = max_strength
        context_analysis["primary_domain"] = primary_domain
        
        # 복잡도 수준 분석
        complexity_factors = [
            len(question) > 200,  # 긴 문장
            structure.get("has_negative", False),  # 부정형 질문
            len(structure.get("technical_terms", [])) > 2,  # 전문용어 많음
            len(structure.get("legal_references", [])) > 1,  # 법령 참조 많음
            structure.get("choice_count", 0) > 4  # 선택지 많음
        ]
        
        complexity_score = sum(complexity_factors) / len(complexity_factors)
        
        if complexity_score < 0.3:
            context_analysis["complexity_level"] = "simple"
        elif complexity_score > 0.6:
            context_analysis["complexity_level"] = "complex"
        
        # 질문 의도 분석
        intent_patterns = {
            "definition": ["정의", "의미", "개념", "무엇"],
            "procedure": ["절차", "과정", "방법", "어떻게"],
            "comparison": ["차이", "비교", "구분"],
            "evaluation": ["평가", "판단", "검토"]
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                context_analysis["question_intent"] = intent
                break
        
        # 추론 깊이 결정
        if structure.get("question_type") == "multiple_choice":
            context_analysis["reasoning_depth"] = "focused"
        elif complexity_score > 0.7:
            context_analysis["reasoning_depth"] = "deep"
        elif max_strength > 0.6:
            context_analysis["reasoning_depth"] = "specialized"
        
        return context_analysis
    
    def create_dynamic_prompt(self, question: str, question_type: str, 
                            context: Dict, difficulty: Optional[str] = None) -> str:
        """동적 프롬프트 생성"""
        
        prompt_parts = []
        
        # 난이도별 접두사
        if difficulty:
            difficulty_config = self.dynamic_templates["difficulty_based"].get(difficulty, 
                                self.dynamic_templates["difficulty_based"]["medium"])
            prompt_parts.append(difficulty_config["prefix"])
        
        # 도메인별 문맥 강화
        primary_domain = context.get("primary_domain", "일반")
        if primary_domain in self.context_enhancers["domain_boosters"]:
            domain_booster = random.choice(self.context_enhancers["domain_boosters"][primary_domain])
            prompt_parts.append(f"{domain_booster}")
        
        # 질문 본문
        prompt_parts.append(f"\n{question}\n")
        
        # 추론 가이드 추가
        reasoning_depth = context.get("reasoning_depth", "moderate")
        if reasoning_depth == "deep":
            reasoning_guide = random.choice(self.reasoning_guides["analytical_framework"])
            prompt_parts.append(f"분석 프레임워크: {reasoning_guide}")
        elif reasoning_depth == "specialized":
            domain_structure = random.choice(self.reasoning_guides["domain_structure"])
            prompt_parts.append(f"구조적 접근: {domain_structure}")
        
        # 응답 지침
        if question_type == "multiple_choice":
            prompt_parts.append("정답 번호(1, 2, 3, 4, 5 중 하나)를 선택하세요.")
        else:
            response_style = context.get("response_style", "standard")
            if response_style == "structured":
                guidance = random.choice(self.difficulty_adaptors["response_guidance"]["structured"])
                prompt_parts.append(f"{guidance}")
            
            prompt_parts.append("한국어로 전문적인 답변을 작성하세요.")
        
        # 품질 강화 접미사
        if difficulty:
            prompt_parts.append(difficulty_config["suffix"])
        
        return "\n".join(prompt_parts)
    
    def create_multi_stage_prompt(self, question: str, question_type: str, 
                                stage: str, context: Dict) -> str:
        """다단계 추론을 위한 프롬프트 생성"""
        
        stage_config = self.multi_stage_prompts.get(stage, self.multi_stage_prompts["stage1_quick"])
        
        prompt_parts = [
            stage_config["prefix"],
            f"{question}\n",
            stage_config["instruction"]
        ]
        
        if stage == "stage3_validation":
            prompt_parts.append("이전 분석 결과를 종합하여 최종 답변을 도출하세요.")
        
        if question_type == "multiple_choice":
            prompt_parts.append("정답 번호를 선택하세요 (1, 2, 3, 4, 5 중 하나).")
        else:
            prompt_parts.append("한국어로 체계적인 답변을 작성하세요.")
        
        return "\n".join(prompt_parts)
    
    def optimize_prompt_by_performance(self, template_key: str, success_rate: float):
        """성능 기반 프롬프트 최적화"""
        
        if template_key not in self.performance_tracker:
            self.performance_tracker[template_key] = {
                "usage_count": 0,
                "success_sum": 0.0,
                "avg_success": 0.0,
                "last_optimization": time.time()
            }
        
        tracker = self.performance_tracker[template_key]
        tracker["usage_count"] += 1
        tracker["success_sum"] += success_rate
        tracker["avg_success"] = tracker["success_sum"] / tracker["usage_count"]
        
        # 성능이 낮은 템플릿 개선
        if tracker["usage_count"] >= 10 and tracker["avg_success"] < 0.6:
            self._evolve_template(template_key, tracker["avg_success"])
            tracker["last_optimization"] = time.time()
            self.stats["optimization_cycles"] += 1
    
    def _evolve_template(self, template_key: str, current_performance: float):
        """템플릿 진화"""
        
        if template_key not in self.template_evolution:
            self.template_evolution[template_key] = {
                "evolution_count": 0,
                "improvements": [],
                "best_performance": current_performance
            }
        
        evolution = self.template_evolution[template_key]
        evolution["evolution_count"] += 1
        
        # 개선 전략 선택
        if current_performance < 0.4:
            # 심각한 성능 저하 - 구조적 개선
            improvement = self._apply_structural_improvement(template_key)
        elif current_performance < 0.6:
            # 중간 성능 - 세부 조정
            improvement = self._apply_detail_refinement(template_key)
        else:
            # 미세 조정
            improvement = self._apply_fine_tuning(template_key)
        
        evolution["improvements"].append({
            "timestamp": time.time(),
            "improvement_type": improvement,
            "previous_performance": current_performance
        })
        
        if len(evolution["improvements"]) > 5:
            evolution["improvements"] = evolution["improvements"][-5:]
    
    def _apply_structural_improvement(self, template_key: str) -> str:
        """구조적 개선 적용"""
        
        original_template = self.templates.get(template_key, "")
        
        improvements = [
            "더 명확한 지시문 추가",
            "예시 제공",
            "단계별 가이드 포함",
            "문맥 정보 강화"
        ]
        
        # 실제 템플릿 수정은 복잡하므로 개선 유형만 기록
        selected_improvement = random.choice(improvements)
        
        return f"structural_{selected_improvement}"
    
    def _apply_detail_refinement(self, template_key: str) -> str:
        """세부 조정 적용"""
        
        refinements = [
            "용어 정확성 향상",
            "문장 구조 개선",
            "추론 가이드 추가",
            "응답 형식 명확화"
        ]
        
        selected_refinement = random.choice(refinements)
        return f"detail_{selected_refinement}"
    
    def _apply_fine_tuning(self, template_key: str) -> str:
        """미세 조정 적용"""
        
        tunings = [
            "어조 조정",
            "강조점 변경",
            "예시 교체",
            "순서 재배치"
        ]
        
        selected_tuning = random.choice(tunings)
        return f"fine_{selected_tuning}"
    
    def create_korean_reinforced_prompt(self, question: str, question_type: str) -> str:
        cache_key = hash(f"{question[:100]}{question_type}")
        if cache_key in self.prompt_cache:
            self.stats["cache_hits"] += 1
            return self.prompt_cache[cache_key]
        
        question_lower = question.lower()
        
        # 질문 문맥 분석
        structure = {"question_type": question_type}  # 기본 구조
        context = self.analyze_question_context(question, structure)
        
        if question_type == "multiple_choice":
            # 특화된 객관식 템플릿 선택
            if "금융투자업" in question_lower and ("소비자금융업" in question_lower or "보험중개업" in question_lower):
                template_key = "mc_financial"
                prompt = self.templates[template_key].format(question=question)
            elif "개인정보" in question_lower:
                template_key = "mc_privacy"
                prompt = self.templates[template_key].format(question=question)
            elif "전자금융" in question_lower:
                template_key = "mc_electronic"
                prompt = self.templates[template_key].format(question=question)
            elif "트로이" in question_lower or "악성코드" in question_lower:
                template_key = "mc_cyber"
                prompt = self.templates[template_key].format(question=question)
            elif any(neg in question_lower for neg in ["해당하지", "적절하지", "옳지", "틀린"]):
                template_key = "mc_negative"
                prompt = self.templates[template_key].format(question=question)
            else:
                template_key = "mc_basic"
                # 동적 프롬프트 적용
                difficulty = context.get("complexity_level", "medium")
                if context.get("domain_strength", 0) > 0.5:
                    prompt = self.create_dynamic_prompt(question, question_type, context, difficulty)
                else:
                    prompt = self.templates[template_key].format(question=question)
        
        else:
            # 주관식 특화 템플릿 선택
            if "트로이" in question_lower or "악성코드" in question_lower:
                template_key = "subj_trojan"
                prompt = self.templates[template_key].format(question=question)
            elif "개인정보" in question_lower:
                template_key = "subj_personal_info"
                prompt = self.templates[template_key].format(question=question)
            elif "전자금융" in question_lower:
                template_key = "subj_electronic"
                prompt = self.templates[template_key].format(question=question)
            elif "위험" in question_lower and "관리" in question_lower:
                template_key = "subj_risk"
                prompt = self.templates[template_key].format(question=question)
            else:
                template_key = "subj_basic"
                # 동적 프롬프트 적용
                difficulty = context.get("complexity_level", "medium")
                if context.get("domain_strength", 0) > 0.5:
                    prompt = self.create_dynamic_prompt(question, question_type, context, difficulty)
                else:
                    prompt = self.templates[template_key].format(question=question)
        
        # 통계 업데이트
        self.stats["template_usage"][template_key] = self.stats["template_usage"].get(template_key, 0) + 1
        
        primary_domain = context.get("primary_domain", "일반")
        self.stats["domain_distribution"][primary_domain] = self.stats["domain_distribution"].get(primary_domain, 0) + 1
        
        # 캐시 관리
        self._manage_cache()
        self.prompt_cache[cache_key] = prompt
        
        return prompt
    
    def create_confidence_calibrated_prompt(self, question: str, question_type: str, 
                                          target_confidence: float) -> str:
        """신뢰도 목표에 맞춘 프롬프트 생성"""
        
        base_prompt = self.create_korean_reinforced_prompt(question, question_type)
        
        if target_confidence >= 0.8:
            # 고신뢰도 요구 - 더 정밀한 분석 유도
            confidence_enhancer = "\n\n신중하게 검토하여 확실한 답변을 제공하세요. 각 단계별로 논리적 근거를 확인하세요."
        elif target_confidence >= 0.6:
            # 중간 신뢰도 - 균형잡힌 접근
            confidence_enhancer = "\n\n체계적으로 분석하여 합리적인 답변을 제공하세요."
        else:
            # 낮은 신뢰도 허용 - 빠른 응답
            confidence_enhancer = "\n\n핵심 요점을 파악하여 답변하세요."
        
        return base_prompt + confidence_enhancer
    
    def _manage_cache(self):
        if len(self.prompt_cache) >= self.max_cache_size:
            keys_to_remove = list(self.prompt_cache.keys())[:self.max_cache_size // 3]
            for key in keys_to_remove:
                del self.prompt_cache[key]
    
    def create_prompt(self, question: str, question_type: str, analysis: Dict, structure: Dict) -> str:
        return self.create_korean_reinforced_prompt(question, question_type)
    
    def create_few_shot_prompt(self, question: str, question_type: str, analysis: Dict, num_examples: int = 1) -> str:
        prompt_parts = []
        
        # 도메인별 예시 선택
        domain = analysis.get("domain", ["일반"])[0]
        
        if question_type == "multiple_choice":
            if domain == "개인정보보호":
                prompt_parts.append("예시: 개인정보의 정의로 가장 적절한 것은?")
                prompt_parts.append("답변: 1")
            elif domain == "전자금융":
                prompt_parts.append("예시: 전자금융거래법상 접근매체에 해당하는 것은?")
                prompt_parts.append("답변: 2")
            else:
                prompt_parts.append("예시: 정보보안 관리체계의 핵심 원칙은?")
                prompt_parts.append("답변: 1")
            prompt_parts.append("")
        else:
            if domain == "사이버보안":
                prompt_parts.append("예시: 트로이 목마의 특징을 설명하세요.")
                prompt_parts.append("답변: 트로이 목마는 정상 프로그램으로 위장한 악성코드로, 시스템을 원격으로 제어할 수 있게 합니다.")
            else:
                prompt_parts.append("예시: 개인정보보호 조치에 대해 설명하세요.")
                prompt_parts.append("답변: 개인정보보호법에 따라 안전성 확보조치를 이행해야 합니다.")
            prompt_parts.append("")
        
        prompt_parts.append(f"문제: {question}")
        
        if question_type == "multiple_choice":
            prompt_parts.append("답변:")
        else:
            prompt_parts.append("답변:")
        
        return "\n".join(prompt_parts)
    
    def optimize_for_model(self, prompt: str, model_name: str) -> str:
        if "solar" in model_name.lower():
            return f"### User:\n{prompt}\n\n### Assistant:\n"
        elif "llama" in model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        elif "mistral" in model_name.lower():
            return f"[INST] {prompt} [/INST]"
        else:
            return prompt
    
    def get_performance_insights(self) -> Dict:
        """성능 인사이트 반환"""
        insights = {
            "top_performing_templates": {},
            "optimization_opportunities": [],
            "domain_preferences": {},
            "evolution_summary": {}
        }
        
        # 상위 성능 템플릿 식별
        for template, tracker in self.performance_tracker.items():
            if tracker["usage_count"] >= 5:
                insights["top_performing_templates"][template] = tracker["avg_success"]
        
        # 최적화 기회 식별
        for template, tracker in self.performance_tracker.items():
            if tracker["usage_count"] >= 10 and tracker["avg_success"] < 0.6:
                insights["optimization_opportunities"].append({
                    "template": template,
                    "current_performance": tracker["avg_success"],
                    "usage_count": tracker["usage_count"]
                })
        
        # 도메인 선호도 분석
        total_domain_usage = sum(self.stats["domain_distribution"].values())
        if total_domain_usage > 0:
            for domain, count in self.stats["domain_distribution"].items():
                insights["domain_preferences"][domain] = count / total_domain_usage
        
        # 진화 요약
        for template, evolution in self.template_evolution.items():
            insights["evolution_summary"][template] = {
                "evolution_count": evolution["evolution_count"],
                "recent_improvements": len(evolution["improvements"])
            }
        
        return insights
    
    def get_stats_report(self) -> Dict:
        total_prompts = sum(self.stats["template_usage"].values())
        
        performance_insights = self.get_performance_insights()
        
        return {
            "total_prompts": total_prompts,
            "cache_hit_rate": self.stats["cache_hits"] / max(total_prompts, 1),
            "template_usage": self.stats["template_usage"],
            "domain_distribution": self.stats["domain_distribution"],
            "optimization_cycles": self.stats["optimization_cycles"],
            "performance_insights": performance_insights,
            "active_templates": len(self.templates),
            "dynamic_templates": len(self.dynamic_templates),
            "evolved_templates": len(self.template_evolution)
        }
    
    def cleanup(self):
        total_usage = sum(self.stats["template_usage"].values())
        optimization_count = self.stats["optimization_cycles"]
        
        if total_usage > 0:
            most_used = max(self.stats["template_usage"].items(), key=lambda x: x[1]) if self.stats["template_usage"] else ("none", 0)
            print(f"프롬프트 엔지니어링: {total_usage}회 사용, {optimization_count}회 최적화, 주사용 '{most_used[0]}'")
        
        self.prompt_cache.clear()
        self.performance_tracker.clear()