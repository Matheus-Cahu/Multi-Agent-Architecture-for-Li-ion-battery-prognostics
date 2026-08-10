const hre = require("hardhat");

async function main() {
  const Anchor = await hre.ethers.getContractFactory("ReportAnchor");
  const anchor = await Anchor.deploy();
  await anchor.waitForDeployment();
  console.log("ReportAnchor em:", await anchor.getAddress());
}

main().catch((e) => { console.error(e); process.exit(1); });
