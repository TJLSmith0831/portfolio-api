
from typing_extensions import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl

class RelevantLink(BaseModel):
    type: str = Field(..., description="Type of page (about, profile, news, etc.)")
    url: str = Field(..., description="Fully-qualified URL")


class RelevantInfoResponse(BaseModel):
    identity: Optional[Dict[str, Any]] = None
    school_context: Optional[Dict[str, Any]] = None
    rankings: Optional[Dict[str, Any]] = None
    latest_season_stats: Optional[Union[Dict[str, Any], str]] = None
    background: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    notable_headlines: List[Dict[str, Any]] = Field(default_factory=list)


class PlayerSearchResult(BaseModel):
    query: str
    found: bool
    profile_url: Optional[HttpUrl] = None
    displayed_name: Optional[str] = None

    def __str__(self) -> str:
        if not self.found:
            return f"[247] '{self.query}' → no results"

        return f"[247] '{self.query}' → {self.displayed_name} ({self.profile_url})"


class PlayerFitRequest(BaseModel):
    player_name: str
    team_name: str
    player_profile: RelevantInfoResponse


class PlayerFitSummary(BaseModel):
    player: str
    team: str
    position: str
    fit_score: int = Field(..., ge=0, le=100)
    scheme_fit: str
    depth_chart_impact: str
    development_outlook: str
    risk_factors: List[Optional[str]]
    overall_summary: str

    def __str__(self) -> str:
        risks = "\n".join(f"  - {risk}" for risk in self.risk_factors) or "  - None"

        return (
            f"------------------\n"
            f"Player Fit Summary\n"
            f"------------------\n"
            f"Player: {self.player}\n"
            f"Team: {self.team}\n"
            f"Position: {self.position}\n"
            f"Fit Score: {self.fit_score}/100\n\n"
            f"Scheme Fit: {self.scheme_fit}\n"
            f"Depth Chart Impact: {self.depth_chart_impact}\n"
            f"Development Outlook: {self.development_outlook}\n\n"
            f"Risk Factors:\n"
            f"{risks}\n\n"
            f"Overall Summary:\n"
            f"{self.overall_summary}"
        )

class PlayerFitSummaryRequest(BaseModel):
    player_name: str
    requested_team_name: str

class PlayerFitSummaryResponse(BaseModel):
    summary: PlayerFitSummary
