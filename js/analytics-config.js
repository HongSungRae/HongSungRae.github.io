/**
 * Private analytics configuration (not linked from the public site).
 *
 * Setup:
 * 1. Create a free account at https://www.goatcounter.com
 * 2. Add site URL: https://hongsungRae.github.io/
 * 3. Copy the count endpoint (Settings → Sites → Code)
 * 4. Create an API key (top menu → API) with read access
 * 5. Fill goatcounterEndpoint and apiToken below
 *
 * Panel password: change panelPasswordHash after running:
 *   echo -n 'YOUR_PASSWORD' | sha256sum
 */
window.SITE_ANALYTICS = {
    goatcounterEndpoint: 'https://sungraehong.goatcounter.com/count',
    apiToken: 'y4k73n3ytbk61e1btvcsvzcchj4e9dsga5cua1ej4zdcvg4yea',
    panelPasswordHash: '952198aa1ba0c69156790dc3d279b0bdb523073e57ba4eb685892d8619e5cb42'
};
