/**
 * Dashboard page.
 */


import { useState } from 'react';
import { Grid2 as Grid, Stack, Box, Typography, Paper } from '@mui/material';
import { useLoaderData } from 'react-router';
import { Line } from 'react-chartjs-2';
import { ChartData } from 'chart.js';

import type { EEGResponse } from '../../services/loaders';


type SelectedChartParams = {
    label: string;
    data: ChartData<"line", number[], unknown>;
};

function SelectedChart({ label, data }: SelectedChartParams): JSX.Element {
    const options = {
        plugins: {
          title: {
            display: true,
            text: label
          },
          legend: {
            display: false
          }
        }
    }

    return (
        <Box >
            <Typography variant="h6" sx={{fontFamily: "monospace"}}>{label}</Typography>
            <Line options={options} data={data} ></Line>
        </Box>
    );
}


function Statistics(): JSX.Element {
    return (
        <></>
    );
}

type DashboardProps = {
    selectedDataset: ChartData<"line", number[], unknown> | null;
}

/**
 * @returns Dashboard component.
 */
export default function Dashboard(): JSX.Element {
    const eegResponse = useLoaderData() as EEGResponse;
    if (eegResponse === null) {
        console.error("Failed to load dashboard data. Redirecting to previous page....");
        window.history.back();
        return (<></>);
    }

    const [dashboardState, _] = useState({
        selectedDataset: null
    } as DashboardProps);

    return (
        <Box
            id="dashboard"
            sx={{
                padding: "10px",
                margin: "10px auto",
                height: "100%",
                width: "80%"
            }}
            component="div">
            <Grid container spacing={2}>
                <Grid>
                    <Paper variant="outlined">
                    </Paper>
                </Grid>
                <Grid>
                    <Stack>
                        <Paper>
                            { dashboardState.selectedDataset && <>
                                <SelectedChart
                                    label={"Current Chart"}
                                    data={dashboardState.selectedDataset}>
                                </SelectedChart>
                                <Statistics />
                            </>}
                        </Paper>
                    </Stack>
                </Grid>
            </Grid>
        </Box>
    );
}
