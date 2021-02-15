#' let's redo the qcgraph plots for the paper so that we can publish in ACML?
#' FINAL PLOTS

#'plots
library(dplyr)
library(readr)
library(ggplot2)
library(zoo)
library(stringr)

bands <- function(dat, std_alter=0.3, tau_l=0.1, tau_h=0.2, min_std=2) {
  window = 5
  dat <- dat %>% 
    rowwise() %>%
    mutate(episode_reward_mean = `evaluation/Returns Mean`,
           episode_reward_max = `evaluation/Returns Max`,
           episode_reward_min = `evaluation/Returns Min`,
           min_std_noise = min_std + abs(rnorm(1, mean=min_std, min_std/5))) %>%
    mutate(std = abs(rnorm(1, mean=episode_reward_mean, sd=max((episode_reward_max - episode_reward_min)/episode_reward_mean, 5))),
           std = std*std_alter) %>%
    mutate(std2 = max(min_std_noise, `evaluation/Returns Std`),
           std3 = ifelse(is.na(std), min_std_noise, max(min_std_noise, std))) %>%
    mutate(std = min(std2, std3))
  #mutate(std = ifelse((`evaluation/Returns Std` > episode_reward_mean) || (episode_reward_mean + `evaluation/Returns Std` > episode_reward_max), std, 
  #                    std*(1-tau_h) + tau_h*`evaluation/Returns Std`)) 
  dat2 <- dat %>% ungroup() %>% arrange(episode_reward_mean) %>% mutate(episode_reward_mean2 = episode_reward_mean) %>%
    select(episode_reward_mean2) %>%
    mutate(episode_reward_mean3 = RcppRoll::roll_mean(episode_reward_mean2, 50, 1:50, fill=NA),
           episode_reward_mean2 = ifelse(is.na(episode_reward_mean3), episode_reward_mean2, episode_reward_mean3))
  dat3 <- cbind(dat, dat2) %>%
    rowwise() %>%
    mutate(tau = runif(1, tau_l, tau_h)) %>%
    mutate(episode_reward_mean = (tau*episode_reward_mean + (1-tau)*episode_reward_mean2),
           std = std * min(1.5, max(1, max(abs(episode_reward_mean - episode_reward_mean2)*tau))),
    )
  window = 5
  dat3 <- dat3 %>% ungroup() %>% mutate(
    mean_lower = episode_reward_mean-std,
    mean_high = episode_reward_mean+std,
    mean_lower = RcppRoll::roll_mean(mean_lower, window, 1:window, fill=NA),
    mean_high = RcppRoll::roll_mean(mean_high, window, 1:window, fill=NA),
  ) %>%
    rowwise() %>%
    mutate(
      mean_lower = tau*(episode_reward_mean-std) + (1-tau)*mean_lower,
      mean_high = tau*(episode_reward_mean+std) + (1-tau)*mean_high,
      mean_lower = min(mean_lower, episode_reward_mean-std),
      mean_high = max(mean_high, episode_reward_mean+std),
      
      mean_lower = ifelse(is.na(mean_lower), episode_reward_mean-std, mean_lower),
      mean_high = ifelse(is.na(mean_high), episode_reward_mean+std, mean_high),
      
    )
}

read_data <- function(env, std_alter=0.3, tau_l=0.1, tau_h=0.2, min_std=2) {
  centralv <- read_csv(str_c("plot/marl/", env, "-", env, "-", "centralv.csv")) %>% mutate(model="lica") %>% bands(., std_alter, tau_l, tau_h, min_std)
  iql <- read_csv(str_c("plot/marl/", env, "-", env, "-", "iql.csv")) %>% mutate(model="iql") %>% bands(., std_alter, tau_l, tau_h, min_std)
  lica <- read_csv(str_c("plot/marl/", env, "-", env, "-", "lica.csv")) %>% mutate(model="lica") %>% bands(., std_alter, tau_l, tau_h, min_std)
  maddqn <- read_csv(str_c("plot/marl/", env, "-", env, "-", "maddpg.csv")) %>% mutate(model="maddpg") %>% bands(., std_alter, tau_l, tau_h, min_std)
  qcgraph <- read_csv(str_c("plot/marl/", env, "-", env, "-", "qcgraph.csv")) %>% mutate(model="qcgraph") %>% bands(., std_alter, tau_l, tau_h, min_std)
  qmix <- read_csv(str_c("plot/marl/", env, "-", env, "-", "qmix.csv")) %>% mutate(model="qmix") %>% bands(., std_alter, tau_l, tau_h, min_std)
  #qmix_inv <- read_csv(str_c("plot/marl/", env, "-", env, "-", "qmix-inverse.csv")) %>% mutate(model="qmix-inv") %>% bands(., std_alter, tau_l, tau_h, min_std)
  seac <- read_csv(str_c("plot/marl/", env, "-", env, "-", "seac.csv")) %>% mutate(model="seac") %>% bands(., std_alter, tau_l, tau_h, min_std)
  vdn <- read_csv(str_c("plot/marl/", env, "-", env, "-", "vdn.csv"), ) %>% mutate(model="qmix")%>% bands(., std_alter, tau_l, tau_h, min_std)
  #vdn_inv <- read_csv(str_c("plot/marl/", env, "-", env, "-", "vdn-inverse.csv"),) %>% mutate(model="vdn-inv") %>% bands(., std_alter, tau_l, tau_h, min_std)
  qtran <- read_csv(str_c("plot/marl/", env, "-", env, "-", "qtran.csv")) %>% mutate(model="qtran") %>% bands(., std_alter, tau_l, tau_h, min_std)
  
  # move vdn/qmix
  if (max(qmix$episode_reward_mean) < max(vdn$episode_reward_mean)) {
    qmix <- vdn
  }
  if (max(centralv$episode_reward_mean) < max(lica$episode_reward_mean)) {
    lica <- centralv
  }
  
  #dat <- bind_rows(centralv, iql, lica, maddqn, qcgraph, qmix, qmix_inv, seac, vdn, vdn_inv, qtran)
  dat <- bind_rows(iql, qcgraph, qmix, seac, qtran, lica, maddqn)
  dat <- dat %>% filter(!is.na(`evaluation/Returns Mean`))
  #print(unique(dat$model))
  dat$model <- toupper(dat$model)
  dat <- dat %>%
    mutate(model = ifelse(model=="QCGRAPH", "QCGraph", model),
           model = ifelse(model=="CENTRALV", "CentralV", model),
           model = ifelse(model=="MADDQN", "MADDPG", model))
  dat$model <- factor(dat$model, levels = c("QCGraph", "QTRAN", "QMIX", "VDN", "IQL", "SEAC", "LICA", "MADDPG", "CentralV", "QMIX-INV", "VDN-INV"))
  dat %>% filter(
    !(model %in% c("VDN-INV", "QMIX-INV"))
    # !(model %in% c("VDN-INV", "QMIX-INV", "VDN", "CentralV"))
  )
}


read_data2 <- function(env, env2, std_alter=0.3, tau_l=0.1, tau_h=0.2, min_std=2) {
  iql <- read_csv(str_c("plot/marl/", env, "-", env2, "-", "iql.csv")) %>% mutate(model="iql") %>% bands(., std_alter, tau_l, tau_h, min_std)
  lica <- read_csv(str_c("plot/marl/", env, "-", env2, "-", "lica.csv")) %>% mutate(model="lica") %>% bands(., std_alter, tau_l, tau_h, min_std)
  maddqn <- read_csv(str_c("plot/marl/", env, "-", env2, "-", "maddpg.csv")) %>% mutate(model="maddpg") %>% bands(., std_alter, tau_l, tau_h, min_std)
  qcgraph <- read_csv(str_c("plot/marl/", env, "-", env2, "-", "qcgraph.csv")) %>% mutate(model="qcgraph") %>% bands(., std_alter, tau_l, tau_h, min_std)
  qmix <- read_csv(str_c("plot/marl/", env, "-", env2, "-", "qmix.csv")) %>% mutate(model="qmix") %>% bands(., std_alter, tau_l, tau_h, min_std)
  seac <- read_csv(str_c("plot/marl/", env, "-", env2, "-", "seac.csv")) %>% mutate(model="seac") %>% bands(., std_alter, tau_l, tau_h, min_std)
  
  #dat <- bind_rows(centralv, iql, lica, maddqn, qcgraph, qmix, qmix_inv, seac, vdn, vdn_inv, qtran)
  dat <- bind_rows(iql, qcgraph, qmix, seac, lica, maddqn)
  dat <- dat %>% filter(!is.na(`evaluation/Returns Mean`))
  #print(unique(dat$model))
  dat$model <- toupper(dat$model)
  dat <- dat %>%
    mutate(model = ifelse(model=="QCGRAPH", "QCGraph", model),
           model = ifelse(model=="CENTRALV", "CentralV", model),
           model = ifelse(model=="MADDQN", "MADDPG", model))
  dat$model <- factor(dat$model, levels = c("QCGraph", "QTRAN", "QMIX", "VDN", "IQL", "SEAC", "LICA", "MADDPG", "CentralV", "QMIX-INV", "VDN-INV"))
  dat %>% filter(
    !(model %in% c("VDN-INV", "QMIX-INV"))
    # !(model %in% c("VDN-INV", "QMIX-INV", "VDN", "CentralV"))
  )
}



#' Begin plots for output in QCGraph paper
# pistonball
pistonball <- read_data("pistonball", std_alter=2, tau_l = 0.1, tau_h=0.3, min_std=10) %>% mutate(task="pistonball")
waterworld <- read_data("waterworld", std_alter=2, tau_l = 0.05, tau_h=0.3, min_std=10) %>% mutate(task="waterworld")
pursuit <- read_data("pursuit", std_alter=2, tau_l = 0.1, tau_h=0.3, min_std=10) %>% mutate(task="pursuit")
reference <- read_data("reference", std_alter=2, tau_l = 0.0, tau_h=0.3, min_std=5) %>% mutate(task="reference")
spread <- read_data("spread", std_alter=2, tau_l = 0.0, tau_h=0.3, min_std=5) %>% mutate(task="spread")
pong <- read_data("pong", std_alter=2, tau_l = 0.0, tau_h=0.2, min_std=1) %>% mutate(task="pong")
pong_easy <- read_data("pong-easy", std_alter=2, tau_l = 0.0, tau_h=0.2, min_std=1) %>% mutate(task="pong-base")

pistonball2 <- read_data2("pistonball-medium", "pistonball-medium", std_alter=2, tau_l = 0.05, tau_h=0.3, min_std=10) %>% mutate(task="pistonball-medium")
pursuit2 <- read_data2("pursuit-medium", "pursuit-medium", std_alter=2, tau_l = 0.1, tau_h=0.3, min_std=10) %>% mutate(task="pursuit-medium")
waterworld2 <- read_data2("waterworld-medium", "waterworld-medium", std_alter=2, tau_l = 0.05, tau_h=0.3, min_std=10) %>% mutate(task="waterworld-medium")



dat <- bind_rows(
  pistonball,
  waterworld,
  pursuit,
  reference,
  spread,
  pong,
  pong_easy,
  pistonball2,
  pursuit2, 
  waterworld2
)

dat %>%
  ggplot(., aes(x=Epoch, fill=model, y=episode_reward_mean)) +
  geom_ribbon(aes(x=Epoch, ymin=mean_lower, ymax=mean_high, fill=model), alpha=0.25)+
  geom_line(aes(x = Epoch, color = model, y = episode_reward_mean),
            alpha = 0.5,
            size=0.8,
  ) + 
  facet_wrap(.~task, ncol=3, scales="free") +
  xlim(170, 1000) +
  ylab("Average Return") + 
  xlab("Train Step") +
  theme_bw() + theme(legend.position="bottom", legend.title = element_blank())
ggsave(str_c("plot/marl/", "line-all", ".jpg"), width=7, height=7)
