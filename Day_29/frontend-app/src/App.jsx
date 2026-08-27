import InspectionHero from "./components/InspectionHero";
import CaseLedger from "./components/CaseLedger";
import CaseNotes from "./components/CaseNotes";
import DeploymentCertificate from "./components/DeploymentCertificate";
import FacilityBlueprint from "./components/FacilityBlueprint";
import SiteFooter from "./components/SiteFooter";

export default function App() {
  return (
    <>
      <InspectionHero />
      <CaseLedger />
      <CaseNotes />
      <DeploymentCertificate />
      <FacilityBlueprint />
      <SiteFooter />
    </>
  );
}
