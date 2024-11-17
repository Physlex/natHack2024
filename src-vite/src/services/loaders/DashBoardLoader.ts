export type EEGResponse = {
    timeseries: Array<number>;
    timestamps: Array<number>;
}

export default async function dashboardLoader(): Promise<EEGResponse[] | null> {
    const eegResponse = await fetch("api/dashboard/eeg/", {method: "GET"});

    if (eegResponse.status >= 400) {
        console.error(`Failed to access eeg api. Reason: ${eegResponse.status}`)
        window.history.back();
        return null;
    }

    const data = await eegResponse.json() as EEGResponse[];
    return (data);
}
