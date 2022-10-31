import { PollUntilValid } from "./pollUntilValid.js";
import { setDownloadLink, setIsBusy, setDisabled } from "../domHelpers.js";

/**
 * This function polls repeatedly data from a URL as long as it receives
 *     HTTP 202 Accepted Codes.
 *     It will stop if it get a 200 OK or any Code outside the range of 200-299.
 *     Depending on Success(200), Pending(202), or any failure (!200-299 or NetworkError) it
 *     will display different messages to the user.
 *
 *     The function assumes the availability of 2 HTML Elements with the following ids:
 *     @example
 *     <span id="YOUR_PREFIX_HERE-status"></span>
 *     <a id="YOUR_PREFIX_HERE-download-button"></a>
 *
 * @param downloadUrl
 * @param prefix
 * @returns {Promise<Response>}
 */
async function poll(url, prefix) {
    function validateFn(response) {
        // valid once the status is 200
        return response.status === 200;
    }

    async function onProgress(response) {
        document.getElementById(`${prefix}-status`).innerText = "Generating, please wait…";
    }

    async function onValid(response) {
        const result = await response.json();
        const { status, href } = result;
        setDownloadLink(`${prefix}-download-button`, href);
        setIsBusy(`${prefix}-download-button`, false);
        setDisabled(`${prefix}-download-button`, false);

        document.getElementById(`${prefix}-status`).innerText = status;
    }

    /**
     * Displays an error message and disappears the download button
     * @param _prefix sketch-map | quality-report
     */
    function handleError(_prefix) {
        // a message text container
        document.getElementById(`${_prefix}-status`).innerHTML = "Sorry! <br>Something went wrong while trying to generate the Result. Please try again later.";
        // a button to disappear
        document.getElementById(`${_prefix}-download-button`).style.display = "none";
    }

    async function onError(response) {
        const errorText = await response.text();
        console.log(response.status, response.statusText, errorText);
        handleError(prefix);
    }

    try {
        return PollUntilValid.poll(url, validateFn, 1000, onValid, onProgress, onError);
    } catch (e) {
        // Network Error or other reason why the request could not be completed
        console.log(e);
        handleError(prefix);
        return null;
    }
}

export {
    poll,
};