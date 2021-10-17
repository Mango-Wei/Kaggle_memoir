{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "acbd3039",
   "metadata": {
    "papermill": {
     "duration": 0.007488,
     "end_time": "2021-08-09T15:24:07.952785",
     "exception": false,
     "start_time": "2021-08-09T15:24:07.945297",
     "status": "completed"
    },
    "tags": []
   },
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "ed5fc6ac",
   "metadata": {
    "papermill": {
     "duration": 0.006173,
     "end_time": "2021-08-09T15:24:07.965660",
     "exception": false,
     "start_time": "2021-08-09T15:24:07.959487",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "This is an ensemble of public notebooks available for this competition as below, kindly upvote and appreciate the original authors for their work. The credit of this notebook goes entirely to them\n",
    "\n",
    "* rerun-seti-e-t-volo-d1-baseline-inference\n",
    "* seti-bl-spatial-info-tf-tpu\n",
    "* seti-bl-tf-starter-tpu\n",
    "* seti-learned-image-resizing\n",
    "* lb-0-980-efficientnet-b0-more-epoch\n",
    "* inference-5x-ensemble-vanilla-resnet34d-seti\n",
    "* one-stop-understanding-eda-efficientnet\n",
    "* old-data-vs-new-data\n",
    "* setietyuta"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c310e373",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-09T15:24:07.984438Z",
     "iopub.status.busy": "2021-08-09T15:24:07.982684Z",
     "iopub.status.idle": "2021-08-09T15:24:07.994761Z",
     "shell.execute_reply": "2021-08-09T15:24:07.994141Z"
    },
    "papermill": {
     "duration": 0.022887,
     "end_time": "2021-08-09T15:24:07.994917",
     "exception": false,
     "start_time": "2021-08-09T15:24:07.972030",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "67878c54",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-09T15:24:08.014561Z",
     "iopub.status.busy": "2021-08-09T15:24:08.013908Z",
     "iopub.status.idle": "2021-08-09T15:24:08.793923Z",
     "shell.execute_reply": "2021-08-09T15:24:08.792763Z",
     "shell.execute_reply.started": "2021-07-18T06:57:14.927101Z"
    },
    "papermill": {
     "duration": 0.792524,
     "end_time": "2021-08-09T15:24:08.794120",
     "exception": false,
     "start_time": "2021-08-09T15:24:08.001596",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data1 = pd.read_csv(\"../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv\") #0.727\n",
    "data2 = pd.read_csv(\"../input/seti-bl-spatial-info-tf-tpu/submission.csv\") #0.673\n",
    "data3 = pd.read_csv(\"../input/seti-bl-tf-starter-tpu/submission.csv\") #0.641\n",
    "data4 = pd.read_csv(\"../input/setietyuta/submissio.csv\") #0.744\n",
    "data5 = pd.read_csv(\"../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv\") #0.750\n",
    "data6 = pd.read_csv(\"../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv\") #0.725\n",
    "data7 = pd.read_csv(\"../input/one-stop-understanding-eda-efficientnet/submission.csv\") #0.705\n",
    "data8 = pd.read_csv(\"../input/old-data-vs-new-data/new_submission.csv\") #0.715"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ee65d3f7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-09T15:24:08.820197Z",
     "iopub.status.busy": "2021-08-09T15:24:08.819245Z",
     "iopub.status.idle": "2021-08-09T15:24:08.839369Z",
     "shell.execute_reply": "2021-08-09T15:24:08.839923Z",
     "shell.execute_reply.started": "2021-07-18T06:57:15.145599Z"
    },
    "papermill": {
     "duration": 0.039208,
     "end_time": "2021-08-09T15:24:08.840131",
     "exception": false,
     "start_time": "2021-08-09T15:24:08.800923",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>000bf832cae9ff1</td>\n",
       "      <td>0.083485</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>000c74cc71a1140</td>\n",
       "      <td>0.067872</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>000f5f9851161d3</td>\n",
       "      <td>0.072294</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>000f7499e95aba6</td>\n",
       "      <td>0.107086</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>00133ce6ec257f9</td>\n",
       "      <td>0.070402</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                id    target\n",
       "0  000bf832cae9ff1  0.083485\n",
       "1  000c74cc71a1140  0.067872\n",
       "2  000f5f9851161d3  0.072294\n",
       "3  000f7499e95aba6  0.107086\n",
       "4  00133ce6ec257f9  0.070402"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "19f64919",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-09T15:24:08.869517Z",
     "iopub.status.busy": "2021-08-09T15:24:08.868460Z",
     "iopub.status.idle": "2021-08-09T15:24:08.872946Z",
     "shell.execute_reply": "2021-08-09T15:24:08.873459Z",
     "shell.execute_reply.started": "2021-07-18T06:57:15.172069Z"
    },
    "papermill": {
     "duration": 0.025037,
     "end_time": "2021-08-09T15:24:08.873655",
     "exception": false,
     "start_time": "2021-08-09T15:24:08.848618",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>000bf832cae9ff1</td>\n",
       "      <td>0.083085</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>000c74cc71a1140</td>\n",
       "      <td>0.079928</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>000f5f9851161d3</td>\n",
       "      <td>0.078726</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>000f7499e95aba6</td>\n",
       "      <td>0.146394</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>00133ce6ec257f9</td>\n",
       "      <td>0.073958</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                id    target\n",
       "0  000bf832cae9ff1  0.083085\n",
       "1  000c74cc71a1140  0.079928\n",
       "2  000f5f9851161d3  0.078726\n",
       "3  000f7499e95aba6  0.146394\n",
       "4  00133ce6ec257f9  0.073958"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c37ade19",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-09T15:24:08.896338Z",
     "iopub.status.busy": "2021-08-09T15:24:08.895606Z",
     "iopub.status.idle": "2021-08-09T15:24:08.918281Z",
     "shell.execute_reply": "2021-08-09T15:24:08.918699Z"
    },
    "papermill": {
     "duration": 0.037365,
     "end_time": "2021-08-09T15:24:08.918860",
     "exception": false,
     "start_time": "2021-08-09T15:24:08.881495",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data9 = data1\n",
    "data9[\"target\"] = 0.75*data5[\"target\"]+0.20*data4[\"target\"] + 0.05*data1[\"target\"] + 0.00*data6[\"target\"] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "db64c4a9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-08-09T15:24:08.937442Z",
     "iopub.status.busy": "2021-08-09T15:24:08.936793Z",
     "iopub.status.idle": "2021-08-09T15:24:09.104470Z",
     "shell.execute_reply": "2021-08-09T15:24:09.103638Z"
    },
    "papermill": {
     "duration": 0.17852,
     "end_time": "2021-08-09T15:24:09.104601",
     "exception": false,
     "start_time": "2021-08-09T15:24:08.926081",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data9.to_csv(\"submission.csv\",index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 9.730743,
   "end_time": "2021-08-09T15:24:09.720718",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2021-08-09T15:23:59.989975",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
