from typing import Dict, Any

class EngineeringTool:
    def stability_factor(self, c: float, phi: float, gamma: float, h: float, beta: float) -> Dict[str, Any]:
        """
        简化边坡稳定性安全系数计算 (瑞典条分法简化或无限边坡模型)
        这里使用无限边坡模型作为示例: Fs = (c + (gamma * h * cos^2(beta) - u) * tan(phi)) / (gamma * h * sin(beta) * cos(beta))
        假设无孔隙水压力 u=0
        """
        import math
        
        try:
            beta_rad = math.radians(beta)
            phi_rad = math.radians(phi)
            
            numerator = c + (gamma * h * (math.cos(beta_rad)**2)) * math.tan(phi_rad)
            denominator = gamma * h * math.sin(beta_rad) * math.cos(beta_rad)
            
            if denominator == 0:
                return {"error": "Denominator is zero"}
                
            fs = numerator / denominator
            return {
                "Fs": round(fs, 3),
                "status": "Stable" if fs > 1.3 else ("Unstable" if fs < 1.0 else "Critical"),
                "params": {"c": c, "phi": phi, "gamma": gamma, "h": h, "beta": beta}
            }
        except Exception as e:
            return {"error": str(e)}

engineering_tool = EngineeringTool()
